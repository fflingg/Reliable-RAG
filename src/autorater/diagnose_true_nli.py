"""Diagnose one failed NLI pair; keep diagnostics separate from benchmark results."""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
import traceback
from pathlib import Path

from autorater_benchmark import load_rows, nli_text


def safe_float(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def label_sequences(tokenizer):
    """SentencePiece may encode '0' as a whitespace token plus a digit token."""
    sequences = {label: tokenizer.encode(label, add_special_tokens=False) for label in ['0', '1']}
    if any(not ids for ids in sequences.values()):
        raise ValueError(f'Empty label encoding: {sequences}')
    return sequences


def teacher_forcing_tokens(label_ids, eos_id, decoder_start_id):
    """Score every label token and EOS, with an explicitly shifted decoder input."""
    targets = list(label_ids) + [eos_id]
    return [decoder_start_id] + targets[:-1], targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--condition-id', default='squad_10820_insuf')
    parser.add_argument('--model', default='google/t5_xxl_true_nli_mixture')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--modes', nargs='+', choices=['int8', 'fp32_cpu'], default=['fp32_cpu'],
                        help='FP32 CPU is the V100-safe default; int8 is only for supported GPUs.')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    rows = load_rows(root / 'data/sufficiencybench_v1.1/val.json')
    failed = next(row for row in rows if row['condition_id'] == args.condition_id)
    # Include the matched sufficient condition as a control.
    pair = [failed] + [row for row in rows if row['question_id'] == failed['question_id']
                      and row['condition_id'] != failed['condition_id']]
    if args.prepare_only:
        print(json.dumps([{'condition_id': row['condition_id'], 'question': row['question'],
                           'gold_answer': row['gold_answer'], 'label': row['label']}
                          for row in pair], indent=2, ensure_ascii=False))
        print('Preparation passed: no model imported or loaded.')
        return

    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
    folder = root / 'results' / 'diagnostics' / f'{args.condition_id}_{time.strftime("%Y%m%d_%H%M%S")}'
    folder.mkdir(parents=True, exist_ok=False)
    report = {'model': args.model, 'requested_revision': args.revision,
              'torch': torch.__version__, 'transformers': transformers.__version__,
              'condition_ids': [row['condition_id'] for row in pair], 'runs': []}

    def save():
        (folder / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                                      allow_nan=False) + '\n', encoding='utf-8')

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    label_ids = label_sequences(tokenizer)
    report['label_token_ids'] = label_ids
    report['label_score_definition'] = 'Teacher-forced sum of log probabilities for all label tokens plus EOS; mean also reported.'
    print(f'Diagnostic output: {folder}', flush=True)
    save()

    def score_details(scores):
        vector = scores.detach().float().cpu().flatten()
        finite = torch.isfinite(vector)
        result = {'nan_count': int(torch.isnan(vector).sum()),
                  'positive_inf_count': int(torch.isposinf(vector).sum()),
                  'negative_inf_count': int(torch.isneginf(vector).sum()),
                  'all_finite': bool(finite.all()),
                  'label_first_token_logits': {label: safe_float(vector[ids[0]]) for label, ids in label_ids.items()},
                  'eos_logit': safe_float(vector[tokenizer.eos_token_id])}
        if finite.all():
            best = torch.topk(vector, k=min(8, vector.numel()))
            result['top_tokens'] = [{'id': int(idx), 'token': tokenizer.convert_ids_to_tokens(int(idx)),
                                     'logit': safe_float(value)} for idx, value in zip(best.indices, best.values)]
        return result

    for mode in args.modes:
        entry = {'mode': mode, 'status': 'loading', 'samples': []}
        report['runs'].append(entry)
        save()
        model = inputs = output = forward = None
        try:
            kwargs = {'revision': args.revision, 'dtype': torch.float32, 'low_cpu_mem_usage': True}
            if mode == 'int8':
                if not torch.cuda.is_available():
                    raise RuntimeError('INT8 diagnostic requires a GPU allocation.')
                if torch.cuda.get_device_capability() < (7, 5):
                    raise RuntimeError('LLM.int8 requires Turing (SM75) or newer; use fp32_cpu on V100.')
                kwargs.update(device_map={'': 0}, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
                entry['gpu'] = torch.cuda.get_device_name()
            else:
                # Full FP32 T5-XXL exceeds V100 GPU memory; use CPU with a 64G RAM job.
                kwargs['device_map'] = {'': 'cpu'}
                torch.set_num_threads(4)
            print(f'Loading {mode}', flush=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model, **kwargs).eval()
            entry.update(status='running', dtype=str(model.dtype),
                         resolved_revision=getattr(model.config, '_commit_hash', None))
            for row in pair:
                sample = {'condition_id': row['condition_id'], 'label': row['label']}
                entry['samples'].append(sample)
                start = time.monotonic()
                inputs = tokenizer(nli_text(row), return_tensors='pt', truncation=False)
                sample['input_tokens'] = inputs['input_ids'].shape[1]
                inputs = {key: value.to(model.device) for key, value in inputs.items()}
                with torch.inference_mode():
                    decoder_ids = torch.tensor([[model.config.decoder_start_token_id]], device=model.device)
                    forward = model(**inputs, decoder_input_ids=decoder_ids)
                    sample['first_step_logits'] = score_details(forward.logits[0, -1])
                    del forward
                    forward = None
                    output = model.generate(**inputs, do_sample=False, max_new_tokens=3,
                                            return_dict_in_generate=True, output_scores=True,
                                            pad_token_id=tokenizer.pad_token_id)
                ids = output.sequences[0].detach().cpu().tolist()
                sample.update(sequence_ids=ids, tokens=tokenizer.convert_ids_to_tokens(ids),
                              decoded_with_special_tokens=tokenizer.decode(ids, skip_special_tokens=False),
                              decoded=tokenizer.decode(ids, skip_special_tokens=True).strip(),
                              generation_step_scores=[score_details(score[0]) for score in output.scores],
                              seconds=round(time.monotonic() - start, 3))
                sample['valid_binary_label'] = sample['decoded'] in ['0', '1']
                # First-token logits alone cannot compare labels with unequal token lengths.
                sample['label_sequence_scores'] = {}
                for label, tokens in label_ids.items():
                    decoder, targets = teacher_forcing_tokens(
                        tokens, tokenizer.eos_token_id, model.config.decoder_start_token_id)
                    with torch.inference_mode():
                        forward = model(**inputs, decoder_input_ids=torch.tensor([decoder], device=model.device))
                        logits = forward.logits[0].float()
                        target_tensor = torch.tensor(targets, device=logits.device)
                        selected = logits.log_softmax(dim=-1).gather(1, target_tensor[:, None]).squeeze(1)
                        sample['label_sequence_scores'][label] = {
                            'target_ids_including_eos': targets,
                            'all_logits_finite': bool(torch.isfinite(logits).all()),
                            'token_log_probabilities': [safe_float(value) for value in selected],
                            'sum_log_probability': safe_float(selected.sum()),
                            'mean_log_probability': safe_float(selected.mean()),
                        }
                    forward = logits = selected = None
                sample['seconds'] = round(time.monotonic() - start, 3)
                save()
                print(json.dumps({'mode': mode, **sample}, ensure_ascii=False, allow_nan=False), flush=True)
                output = inputs = None
            entry['status'] = 'complete'
        except Exception as exc:
            entry.update(status='error', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
            print(entry['traceback'], flush=True)
        finally:
            model = inputs = output = forward = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            save()
    print(f'Done. Read {folder / "report.json"}. Benchmark predictions were not changed.', flush=True)


if __name__ == '__main__':
    main()
