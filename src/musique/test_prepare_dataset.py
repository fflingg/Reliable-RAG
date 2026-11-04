import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from prepare_dataset import (choose_sample, render_context, truncate_context, read_spec, export_dataset,
                             write_jsonl, parse_label, repair_labels, read_labels, select_pending)
from autorater_benchmark import qwen_messages


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)

    def decode(self, tokens, skip_special_tokens=False):
        return ''.join(tokens)


class DatasetPreparationTests(unittest.TestCase):
    def test_parse_whitespace_without_guessing_from_prose(self):
        self.assertEqual(parse_label('{" sufficient": 0}'), 0)
        for text in ['Reasoning: {"sufficient": 1}', '{"sufficient": true}',
                     '{" sufficient": 0, "sufficient ": 1}', '[["sufficient", 1]]']:
            with self.assertRaises(ValueError):
                parse_label(text)

    def test_repair_is_append_only_idempotent_and_retry_skips_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data, results = base / 'data', base / 'results'
            data.mkdir()
            results.mkdir()
            rows = [{'id': name, 'question': '?', 'context': 'text'} for name in ['good', 'repair', 'retry']]
            write_jsonl(data / 'prepared.jsonl', rows)
            checkpoint = results / 'labels.jsonl'
            write_jsonl(checkpoint, [
                {'id': 'good', 'status': 'ok', 'sufficiency_label': 1, 'raw_output': '{"sufficient": 1}'},
                {'id': 'repair', 'status': 'invalid', 'sufficiency_label': None, 'raw_output': '{" sufficient": 0}'},
                {'id': 'retry', 'status': 'invalid', 'sufficiency_label': None, 'raw_output': 'unfinished reasoning'},
            ])
            original = checkpoint.read_bytes()
            args = SimpleNamespace(dataset_dir=data, results_dir=results, seed=42)
            repair_labels(args)
            self.assertTrue(checkpoint.read_bytes().startswith(original))
            latest = read_labels(checkpoint)
            self.assertEqual(latest['repair']['sufficiency_label'], 0)
            self.assertEqual([r['id'] for r in select_pending(rows, latest, True)], ['retry'])
            self.assertEqual(select_pending(rows, latest), [])
            size = checkpoint.stat().st_size
            repair_labels(args)
            self.assertEqual(checkpoint.stat().st_size, size)

    def test_jsonl_bytes_are_portable_between_windows_and_linux(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prepared.jsonl'
            rows = [{'id': 'q1', 'context': '段落\nnext paragraph'}]
            write_jsonl(path, rows)
            expected = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode('utf-8')
            self.assertEqual(path.read_bytes(), expected)

    def test_sample_is_reproducible_and_ignores_file_order(self):
        rows = [{'id': str(i)} for i in range(1000)]
        self.assertEqual(choose_sample(rows), choose_sample(list(reversed(rows))))
        self.assertEqual(len({row['id'] for row in choose_sample(rows)}), 500)

    def test_keep_all_candidate_paragraphs_and_no_tags(self):
        row = {'id': 'example', 'paragraphs': [
            {'title': f'Title-{i}', 'paragraph_text': f'Body-{i}', 'is_supporting': i == 1}
            for i in range(20)]}
        context = render_context(row)
        self.assertEqual(context.count('[Document '), 20)
        self.assertIn('Body-19', context)
        self.assertNotIn('is_supporting', context)
        self.assertLess(context.index('Body-0'), context.index('Body-19'))
        # Some official records have 18/19 paragraphs; keep them, do not drop/pad.
        row['paragraphs'] = row['paragraphs'][:18]
        self.assertEqual(render_context(row).count('[Document '), 18)

    def test_truncate_and_keep_short_text_exactly(self):
        self.assertEqual(truncate_context('abcdef', CharacterTokenizer(), 4), ('abcd', 6, 4, True))
        self.assertEqual(truncate_context('ab', CharacterTokenizer(), 4), ('ab', 2, 2, False))

    def test_rater_does_not_receive_answers_or_source_labels(self):
        row = {'question': 'Where?', 'context': 'A passage', 'answer': 'SECRET',
               'source_answerable': True, 'question_decomposition': ['SECRET']}
        messages = qwen_messages(row, read_spec()['demonstration'])
        text = json.dumps(messages)
        self.assertNotIn('SECRET', text)
        self.assertNotIn('source_answerable', text)
        self.assertEqual(len(messages), 4)

    def test_invalid_label_is_not_forced_to_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            dataset = base / 'data'
            results = base / 'results'
            dataset.mkdir()
            results.mkdir()
            row = {'id': 'q1', 'question': 'Where?', 'context': 'A passage', 'answer': 'SECRET'}
            write_jsonl(dataset / 'prepared.jsonl', [row])
            write_jsonl(results / 'labels.jsonl', [{'id': 'q1', 'status': 'invalid', 'raw_output': '', 'sufficiency_label': None}])
            summary = export_dataset(SimpleNamespace(dataset_dir=dataset, results_dir=results, seed=42))
            merged = json.loads((results / 'labelled_dataset.jsonl').read_text())
            self.assertIsNone(merged['sufficiency_label'])
            self.assertFalse(summary['ready_for_analysis'])
            self.assertEqual(summary['invalid'], 1)
            sheet = results / 'human_review_blind.csv'
            self.assertNotIn('SECRET', sheet.read_text(encoding='utf-8-sig'))
            sheet.write_text('human edits', encoding='utf-8')
            export_dataset(SimpleNamespace(dataset_dir=dataset, results_dir=results, seed=42))
            self.assertEqual(sheet.read_text(), 'human edits')


if __name__ == '__main__':
    unittest.main()
