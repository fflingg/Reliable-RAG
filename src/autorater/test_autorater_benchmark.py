"""CPU-only protocol checks; these are not model accuracy measurements."""
import json
import tempfile
import unittest
from pathlib import Path
from autorater_benchmark import binary_metrics, parse_qwen, qwen_messages, nli_text, summarize, parser, classify_output


class ProtocolChecks(unittest.TestCase):
    def test_invalid_output_is_incorrect_for_both_classes(self):
        result = binary_metrics([1, 0, 1, 0], [1, 0, None, None], invalid_as_wrong=True)
        for metric in ['F1 Score', 'Accuracy', 'Precision', 'Recall']:
            self.assertEqual(result[metric], 0.5)
        self.assertEqual(result['FP'], 1)
        self.assertEqual(result['FN'], 1)
        self.assertEqual(classify_output('', 'nli')['status'], 'invalid')
        self.assertIsNone(classify_output('', 'nli')['prediction'])
        self.assertEqual(classify_output('not json', 'qwen')['status'], 'invalid')

    def test_legacy_parse_error_is_completed_but_runtime_error_is_not(self):
        rows = [{'condition_id': 'a', 'label': 1}, {'condition_id': 'b', 'label': 0}]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder / 'true_nli.jsonl'
            records = [dict(condition_id='a', status='ok', prediction=1),
                       dict(condition_id='b', status='error', prediction=None, raw_output='',
                            error="ValueError: Unexpected TRUE-NLI label: ''")]
            path.write_text(''.join(json.dumps(r) + '\n' for r in records))
            result = summarize(folder, rows, ['true_nli'])[0]
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['Accuracy'], 0.5)
            self.assertEqual(result['Precision'], 0.5)
            self.assertEqual(result['Recall'], 1.0)
            self.assertEqual(result['invalid_outputs'], 1)
            records[-1].update(raw_output=None, error='RuntimeError: CUDA out of memory')
            path.write_text(''.join(json.dumps(r) + '\n' for r in records))
            result = summarize(folder, rows, ['true_nli'])[0]
            self.assertEqual(result['status'], 'incomplete')
            self.assertIsNone(result['Accuracy'])

    def test_one_method_per_job_and_full_split_default(self):
        args = parser().parse_args(['--method', 'qwen_0shot'])
        self.assertEqual(args.method, 'qwen_0shot')
        self.assertEqual(args.limit_questions, 0)
        self.assertEqual(args.split, 'validation')

    def test_metrics(self):
        scores = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
        for key in ['F1 Score', 'Accuracy', 'Precision', 'Recall']:
            self.assertEqual(scores[key], 0.5)
        self.assertEqual(binary_metrics([1, 0], [0, 0])['F1 Score'], 0)
        with self.assertRaises(ValueError):
            binary_metrics([1, 0], [1, None])

    def test_parser_ignores_thinking_and_rejects_invalid(self):
        self.assertEqual(parse_qwen('<think>{"sufficient": 0}</think>{"sufficient": 1}'), 1)
        for raw in ['yes', '{"sufficient": true}', '{"sufficient": 2}', '{}']:
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                parse_qwen(raw)

    def test_qwen_inputs_exclude_answer_and_label(self):
        row = {'question': 'Where?', 'context': 'A passage.', 'gold_answer': 'SECRET', 'label': 1}
        self.assertNotIn('SECRET', json.dumps(qwen_messages(row)))
        self.assertEqual(len(qwen_messages(row)), 2)
        self.assertEqual(len(qwen_messages(row, row)), 4)
        self.assertIn('SECRET', nli_text(row))

    def test_incomplete_run_does_not_receive_scores(self):
        rows = [{'condition_id': 'a', 'label': 1}, {'condition_id': 'b', 'label': 0}]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder / 'qwen_0shot.jsonl'
            path.write_text(json.dumps({'condition_id': 'a', 'status': 'ok', 'prediction': 1}) + '\n')
            self.assertIsNone(summarize(folder, rows)[0]['Accuracy'])
            with path.open('a') as stream:
                stream.write(json.dumps({'condition_id': 'b', 'status': 'ok', 'prediction': 0}) + '\n')
            self.assertEqual(summarize(folder, rows)[0]['Accuracy'], 1)
            single = summarize(folder, rows, ['qwen_0shot'])
            self.assertEqual(len(single), 1)
            self.assertEqual(single[0]['autorater'], 'qwen_0shot')


if __name__ == '__main__':
    unittest.main()
