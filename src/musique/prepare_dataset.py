"""Prepare MuSiQue-Ans contexts and label them with the selected Qwen autorater."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'autorater'))
from autorater_benchmark import INSTRUCTION, PROMPT_VERSION, parse_qwen, qwen_messages

SOURCE_ID = '1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h'
DEV_FILE = 'musique_ans_v1.0_dev.jsonl'


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError(f'Duplicate IDs in {path}')
    return rows


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.part')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(path)


def choose_sample(rows, count=500, seed=42):
    if count <= 0 or count > len(rows):
        raise ValueError(f'Cannot select {count} questions from {len(rows)}.')
    # Sort before sampling so file ordering alone cannot change selected IDs.
    return random.Random(seed).sample(sorted(rows, key=lambda row: row['id']), count)


def render_context(row):
    """Preserve all candidate paragraphs in source order, including distractors."""
    paragraphs = row['paragraphs']
    if not paragraphs:
        raise ValueError(f'{row["id"]}: no candidate paragraphs')
    return '\n\n'.join(f'[Document {i + 1}] {p["title"]}\n{p["paragraph_text"]}'
                        for i, p in enumerate(paragraphs))


def truncate_context(context, tokenizer, limit):
    if limit < 1:
        raise ValueError('Context limit must be positive.')
    tokens = tokenizer.encode(context, add_special_tokens=False)
    if len(tokens) <= limit:
        return context, len(tokens), len(tokens), False
    retained = tokens[:limit]
    # Re-encoding the saved text is authoritative, including boundary effects.
    while retained:
        text = tokenizer.decode(retained, skip_special_tokens=False)
        actual = len(tokenizer.encode(text, add_special_tokens=False))
        if actual <= limit:
            return text, len(tokens), actual, True
        retained = retained[:-max(1, actual - limit)]
    raise ValueError('Could not produce a non-empty bounded context.')


def spec_path():
    return Path(__file__).with_name('rater_spec.json')


def read_spec():
    spec = json.loads(spec_path().read_text(encoding='utf-8'))
    if spec['instruction'] != INSTRUCTION or spec['prompt_version'] != PROMPT_VERSION:
        raise ValueError('Autorater implementation differs from frozen prompt; review before proceeding.')
    return spec


def download(args):
    raw = args.root / 'data/musique/raw'
    raw.mkdir(parents=True, exist_ok=True)
    target = raw / DEV_FILE
    if target.exists():
        print(f'Already present: {target}; {len(read_jsonl(target))} questions')
        return
    archive = args.archive or raw / 'musique_v1.0.zip'
    if not archive.exists():
        import gdown
        temporary = archive.with_suffix('.zip.part')
        result = gdown.download(id=SOURCE_ID, output=str(temporary), quiet=False)
        if result is None:
            raise RuntimeError('Official download failed. Download the official ZIP manually and use --archive.')
        temporary.replace(archive)
    # Extract only the requested dev member; never extract arbitrary ZIP paths.
    with zipfile.ZipFile(archive) as source:
        members = [name for name in source.namelist() if name.rsplit('/', 1)[-1] == DEV_FILE and not name.startswith('__MACOSX/')]
        if len(members) != 1:
            raise ValueError(f'Expected one {DEV_FILE} member, got {members}')
        payload = source.read(members[0])
        parsed = [json.loads(line) for line in payload.decode('utf-8').splitlines() if line.strip()]
        if not parsed or not all('paragraphs' in row and 'answer' in row for row in parsed):
            raise ValueError('Unexpected MuSiQue format.')
    temporary = target.with_suffix('.jsonl.part')
    temporary.write_bytes(payload)
    temporary.replace(target)
    write_json(raw / 'download_manifest.json', {
        'source': 'https://github.com/StonyBrookNLP/musique', 'drive_file_id': SOURCE_ID,
        'archive_sha256': sha_file(archive), 'dev_sha256': sha_file(target), 'questions': len(parsed),
        'license': 'CC BY 4.0',
    })
    print(f'Downloaded {len(parsed)} questions: {target}', flush=True)


def prepare(args):
    raw = args.root / 'data/musique/raw' / DEV_FILE
    selected = choose_sample(read_jsonl(raw), args.count, args.seed)
    if any(row.get('answerable') is not True for row in selected):
        raise ValueError('Use MuSiQue-Ans with original answerable=True, not MuSiQue-Full.')
    spec = read_spec()
    output = args.dataset_dir
    existing_manifest = output / 'manifest.json'
    requested_revision = args.revision
    if existing_manifest.exists() and requested_revision == 'main':
        requested_revision = json.loads(existing_manifest.read_text(encoding='utf-8'))['model_revision']
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    # Download only config/tokenizer files, not the model weights, and pin the SHA.
    snapshot = Path(snapshot_download(spec['model'], revision=requested_revision, allow_patterns=[
        'config.json', 'tokenizer*', 'vocab*', 'merges*', 'special_tokens_map.json',
        'added_tokens.json', 'chat_template*', '*.model',
    ]))
    revision = snapshot.name
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError(f'Could not identify model commit from snapshot: {snapshot}')
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in selected:
        if row['question'] == spec['demonstration']['question']:
            raise ValueError('Demonstration question overlaps the target sample; review the setup.')
        full = render_context(row)
        if full == spec['demonstration']['context']:
            raise ValueError('Demonstration context overlaps the target sample.')
        context, original_tokens, context_tokens, truncated = truncate_context(full, tokenizer, args.context_limit)
        rows.append({
            'id': row['id'], 'question': row['question'], 'answer': row['answer'],
            'answer_aliases': row.get('answer_aliases', []), 'context': context,
            'context_tokens': context_tokens, 'original_context_tokens': original_tokens,
            'truncated': truncated, 'n_paragraphs': len(row['paragraphs']),
            'source_answerable': row['answerable'], 'context_sha256': hashlib.sha256(context.encode()).hexdigest(),
        })
    manifest = {'dataset': 'MuSiQue-Ans v1.0 dev', 'raw_sha256': sha_file(raw),
                'sample_count': args.count, 'seed': args.seed, 'context_limit': args.context_limit,
                'paragraph_order': 'original array order; all provided candidates (normally 20); title and text',
                'paragraph_count_distribution': dict(sorted(Counter(str(row['n_paragraphs']) for row in rows).items())),
                'tokenizer': spec['model'], 'model_revision': revision, 'rater_spec': spec,
                'sample_ids': [row['id'] for row in rows], 'truncated_count': sum(row['truncated'] for row in rows),
                'context_length_min': min(row['context_tokens'] for row in rows),
                'context_length_max': max(row['context_tokens'] for row in rows)}
    target = output / 'prepared.jsonl'
    # Freeze actual text and order as well as sampling settings.
    encoded = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode('utf-8')
    manifest['prepared_sha256'] = hashlib.sha256(encoded).hexdigest()
    if (output / 'manifest.json').exists():
        previous = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        if previous != manifest or sha_file(target) != manifest['prepared_sha256']:
            raise ValueError('Prepared dataset differs from frozen files. Use another --dataset-dir.')
        print('Prepared dataset unchanged; reusing frozen files.', flush=True)
        return
    write_jsonl(target, rows)
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'sample_ids.json', manifest['sample_ids'])
    print(f'Prepared {len(rows)} contexts, {manifest["truncated_count"]} truncated: {output}', flush=True)


def read_labels(path):
    latest = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                item = json.loads(line)
                latest[item['id']] = item
    return latest


def parse_label(raw):
    """Accept only the established JSON format, allowing whitespace in its key."""
    try:
        return parse_qwen(raw)
    except ValueError:
        text = raw.rsplit('</think>', 1)[-1].strip()
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text).strip()
        pairs = json.loads(text, object_pairs_hook=lambda items: items)
        if (isinstance(pairs, list) and len(pairs) == 1
                and isinstance(pairs[0], tuple) and pairs[0][0].strip() == 'sufficient'
                and type(pairs[0][1]) is int and pairs[0][1] in (0, 1)):
            return pairs[0][1]
        raise ValueError('Expected JSON with integer sufficient=0 or 1.')


def repair_labels(args):
    """Append format-only repairs; preserve every original output and record."""
    checkpoint = args.results_dir / 'labels.jsonl'
    repaired = 0
    with checkpoint.open('a', encoding='utf-8') as stream:
        for item in read_labels(checkpoint).values():
            if item.get('status') != 'invalid':
                continue
            try:
                value = parse_label(item.get('raw_output') or '')
            except ValueError:
                continue
            record = {**item, 'status': 'ok', 'sufficiency_label': value,
                      'repair': 'JSON key whitespace normalization', 'previous_status': 'invalid'}
            record.pop('error', None)
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            repaired += 1
    print(f'Format-only repairs: {repaired}', flush=True)
    export_dataset(args)


def select_pending(rows, latest, retry_invalid=False):
    if retry_invalid:
        return [row for row in rows if latest.get(row['id'], {}).get('status') == 'invalid']
    return [row for row in rows if latest.get(row['id'], {}).get('status') not in ('ok', 'invalid')]


def export_dataset(args):
    rows = read_jsonl(args.dataset_dir / 'prepared.jsonl')
    labels = read_labels(args.results_dir / 'labels.jsonl')
    merged = []
    for row in rows:
        label = labels.get(row['id'], {})
        merged.append({**row, 'sufficiency_label': label.get('sufficiency_label'),
                       'rater_status': label.get('status', 'pending'), 'rater_raw_output': label.get('raw_output')})
    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.results_dir / 'labelled_dataset.jsonl', merged)
    statuses = Counter(row['rater_status'] for row in merged)
    valid = [row for row in merged if row['rater_status'] == 'ok']
    summary = {'total': len(rows), 'valid': len(valid), 'sufficient': sum(row['sufficiency_label'] == 1 for row in valid),
               'insufficient': sum(row['sufficiency_label'] == 0 for row in valid),
               'invalid': statuses['invalid'], 'runtime_errors': statuses['error'], 'pending': statuses['pending'],
               'ready_for_analysis': len(valid) == len(rows), 'label_source': 'Qwen autorater, not human gold',
               'sufficient_fraction_among_valid': sum(row['sufficiency_label'] == 1 for row in valid) / len(valid) if valid else None}
    write_json(args.results_dir / 'summary.json', summary)
    write_jsonl(args.results_dir / 'invalid_review.jsonl', [
        {key: row[key] for key in ['id', 'question', 'context', 'rater_status', 'rater_raw_output']}
        for row in merged if row['rater_status'] in ('invalid', 'error')])
    # Blind random audit: the first 50 IDs are selected independently of predicted labels.
    audit_ids = {r['id'] for r in random.Random(args.seed).sample(rows, min(50, len(rows)))}
    # Include invalid/error records too; leave all human labels genuinely blank.
    audit = [r for r in merged if r['id'] in audit_ids or r['rater_status'] in ('invalid', 'error')]
    review = args.results_dir / 'human_review_blind.csv'
    # Never overwrite an existing annotation sheet containing human work.
    if not review.exists():
        with review.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['id', 'question', 'context', 'random_audit', 'human_label', 'notes'])
            writer.writeheader()
            for row in audit:
                writer.writerow({'id': row['id'], 'question': row['question'], 'context': row['context'],
                                 'random_audit': int(row['id'] in audit_ids), 'human_label': '', 'notes': ''})
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def label(args):
    manifest = json.loads((args.dataset_dir / 'manifest.json').read_text(encoding='utf-8'))
    prepared = args.dataset_dir / 'prepared.jsonl'
    if sha_file(prepared) != manifest['prepared_sha256']:
        raise ValueError('Prepared contexts changed after tokenization.')
    spec = read_spec()
    if manifest['rater_spec'] != spec:
        raise ValueError('Prepared data uses a different rater specification.')
    rows = read_jsonl(prepared)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    config = {'manifest_sha256': sha_file(args.dataset_dir / 'manifest.json'), 'rater': spec,
              'resolved_model_revision': manifest['model_revision'], 'max_input_tokens': args.max_input_tokens,
              'max_new_tokens': 64, 'seed': manifest['seed'],
              'invalid_policy': 'null label; manual review required, never force to 0/1'}
    config_path = args.results_dir / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
        raise ValueError('Existing labels use a different configuration. Use another --results-dir.')
    write_json(config_path, config)
    checkpoint = args.results_dir / 'labels.jsonl'
    latest = read_labels(checkpoint)
    if set(latest) - {row['id'] for row in rows}:
        raise ValueError('Checkpoint contains unexpected question IDs.')
    by_id = {row['id']: row for row in rows}
    for item in latest.values():
        if item['context_sha256'] != by_id[item['id']]['context_sha256']:
            raise ValueError('Checkpoint context hash differs from prepared text.')
        if item['status'] == 'ok' and (type(item['sufficiency_label']) is not int or item['sufficiency_label'] not in (0, 1)):
            raise ValueError('Checkpoint marks a non-binary label as valid.')
    retry_invalid = args.retry_invalid
    pending = select_pending(rows, latest, retry_invalid)
    max_new_tokens = args.retry_max_new_tokens if retry_invalid else 64
    if not pending:
        export_dataset(args)
        return
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError('Run label inside an allocated GPU compute job.')
    torch.manual_seed(manifest['seed'])
    revision = manifest['model_revision']
    tokenizer = AutoTokenizer.from_pretrained(spec['model'], revision=revision)
    print(f'Loading {spec["model"]} at {revision}; remaining {len(pending)}/{len(rows)}', flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        spec['model'], revision=revision, device_map={'': 0}, dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                             bnb_4bit_compute_dtype=torch.float16)).eval()
    write_json(args.results_dir / 'environment.json', {'torch': torch.__version__, 'transformers': transformers.__version__,
               'gpu': torch.cuda.get_device_name(), 'revision': revision, 'compute_dtype': 'float16', 'weights': 'NF4'})
    with checkpoint.open('a', encoding='utf-8', buffering=1) as stream:
        for i, row in enumerate(pending, 1):
            result = {'id': row['id'], 'context_sha256': row['context_sha256'],
                      'sufficiency_label': None, 'raw_output': None, 'status': 'error'}
            if retry_invalid:
                result.update(retry=True, previous_status='invalid', max_new_tokens=max_new_tokens,
                              attempt=latest[row['id']].get('attempt', 1) + 1,
                              retry_policy='same prompt, greedy; increased output budget')
            start = time.monotonic()
            try:
                # qwen_messages deliberately reads only question/context, not answer or source tags.
                messages = qwen_messages(row, spec['demonstration'])
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                      enable_thinking=False)
                inputs = tokenizer(text, add_special_tokens=False, return_tensors='pt', truncation=False)
                result['input_tokens'] = inputs['input_ids'].shape[1]
                if result['input_tokens'] + max_new_tokens > args.max_input_tokens:
                    raise ValueError('Whole prompt exceeds budget; no silent second truncation.')
                inputs = {key: value.to(model.device) for key, value in inputs.items()}
                with torch.inference_mode():
                    generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                               pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
                tokens = generated[0, inputs['input_ids'].shape[1]:].cpu().tolist()
                raw = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                result.update(raw_output=raw, output_token_ids=tokens)
                try:
                    result.update(sufficiency_label=parse_label(raw), status='ok')
                except ValueError as exc:
                    result.update(status='invalid', error=str(exc))
            except Exception as exc:
                result['error'] = f'{type(exc).__name__}: {exc}'
            result['seconds'] = round(time.monotonic() - start, 3)
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f'{i}/{len(pending)} {row["id"]}: {result["status"]}, label={result["sufficiency_label"]}', flush=True)
            if result['status'] == 'error':
                export_dataset(args)
                raise RuntimeError(result['error'])
    export_dataset(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['download', 'prepare', 'label', 'export', 'repair', 'all'])
    parser.add_argument('--retry-invalid', action='store_true', help='Retry only latest invalid records; preserve valid labels')
    parser.add_argument('--retry-max-new-tokens', type=int, default=512, help='Output budget for invalid retries only')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--count', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--context-limit', type=int, default=6000)
    parser.add_argument('--max-input-tokens', type=int, default=8192)
    parser.add_argument('--revision', default='main')
    parser.add_argument('--archive', type=Path, help='Optional already downloaded official ZIP')
    parser.add_argument('--dataset-dir', type=Path)
    parser.add_argument('--results-dir', type=Path)
    args = parser.parse_args()
    if args.retry_max_new_tokens < 1:
        parser.error('--retry-max-new-tokens must be positive')
    args.root = args.root.resolve()
    tag = f'dev{args.count}_seed{args.seed}_ctx{args.context_limit}'
    args.dataset_dir = (args.dataset_dir or args.root / 'data/musique/processed' / tag).resolve()
    args.results_dir = (args.results_dir or args.root / 'results/musique' / tag / 'qwen3.8_1shot').resolve()
    if args.stage == 'all':
        if not (args.root / 'data/musique/raw' / DEV_FILE).exists():
            download(args)
        prepare(args)
        label(args)
    else:
        {'download': download, 'prepare': prepare, 'label': label, 'export': export_dataset,
         'repair': repair_labels}[args.stage](args)


if __name__ == '__main__':
    main()
