import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rag_behavior import (prepare, dataset, read_rows, read_json, write_rows, gen_messages,
                          judge_messages, parse_judgment, final_answer, lexical_decision,
                          response_digest, report, freeze)
import rag_behavior as behavior


class BehaviorTests(unittest.TestCase):
    def test_generation_has_no_explicit_abstention_hint(self):
        prompt = json.dumps(gen_messages({'question': 'Where?', 'context': 'Evidence'})).lower()
        self.assertNotIn("i don't know", prompt)
        self.assertNotIn('cannot determine', prompt)
        self.assertNotIn('abstain', prompt)

    def test_generation_evaluation_resume_and_capped_answer_exclusion(self):
        import hashlib
        class FakeEngine:
            calls = 0
            def __init__(self, *args):
                self.torch = SimpleNamespace(manual_seed=lambda seed: None)
            def infer(self, messages, budget, maximum):
                FakeEngine.calls += 1
                text = messages[0]['content']
                if text.startswith(behavior.JUDGE_INSTRUCTION):
                    return {'raw_output': '{"decision":"incorrect","explanation":"Wrong place"}', 'truncated': False}
                capped = 'CAP_TEST' in text
                return {'raw_output': 'The answer is: Paris', 'truncated': capped}
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{'id': str(i), 'question': 'CAP_TEST' if i else 'Where?', 'context': 'C', 'answer': 'London',
                     'context_sha256': hashlib.sha256(b'C').hexdigest(), 'rater_status': 'ok',
                     'sufficiency_label': i} for i in range(2)]
            source = root / 'source.jsonl'
            write_rows(source, rows)
            args = SimpleNamespace(source=source, dataset_dir=root / 'data', results_dir=root / 'results',
                                   limit=None, seed=42, model='mistral', max_input_tokens=16384,
                                   max_new_tokens=512, judge_max_new_tokens=256, revision='main',
                                   judge_revision='main', retry_invalid=False)
            prepare(args)
            with patch.object(behavior, 'Engine', FakeEngine), patch.object(behavior, 'pin_revision', return_value='frozen'), patch.object(behavior, 'environment', return_value={}):
                behavior.generate(args)
                behavior.evaluate(args)
                self.assertEqual(FakeEngine.calls, 3)
                behavior.evaluate(args)
                self.assertEqual(FakeEngine.calls, 3)  # Valid evaluation resumes without another inference.
            report(args)
            merged = read_rows(args.results_dir / 'mistral/categorized.jsonl')
            self.assertEqual(merged[0]['category'], 'hallucinate')
            self.assertEqual(merged[1]['generation_status'], 'invalid')
            self.assertIsNone(merged[1]['category'])

    def test_prompts_do_not_leak_sufficiency_or_reference_to_generator(self):
        row = {'question': 'Where?', 'context': 'Some evidence', 'answer': 'SECRET_ANSWER',
               'answer_aliases': ['SECRET_ALIAS'], 'sufficiency_label': 'SECRET_LABEL'}
        prompt = json.dumps(gen_messages(row))
        self.assertNotIn('SECRET', prompt)
        judge = json.dumps(judge_messages(row, 'The answer is: Somewhere'))
        self.assertIn('SECRET_ANSWER', judge)
        self.assertNotIn('SECRET_LABEL', judge)
        self.assertNotIn('Some evidence', judge)

    def test_final_answer_equality_does_not_score_explanation_substrings(self):
        raw = 'Paris was mentioned, but I think London. The answer is: London'
        self.assertEqual(final_answer(raw), 'London')
        self.assertIsNone(lexical_decision(final_answer(raw), ['Paris']))
        self.assertEqual(lexical_decision('The Eiffel Tower.', ['Eiffel Tower']), 'perfect')
        self.assertEqual(lexical_decision("I don't know", ['Paris']), 'missing')
        self.assertIsNone(lexical_decision("I don't know for sure, but Paris", ['Paris']))

    def test_evaluator_rejects_invalid_output_instead_of_hallucinate(self):
        for raw in ['', 'hallucinate', '{"decision":"unknown"}', '[]']:
            with self.assertRaises(ValueError):
                parse_judgment(raw)
        self.assertEqual(parse_judgment('{"decision":"acceptable"}')['decision'], 'acceptable')

    def test_valid_cohort_is_frozen_and_keeps_context_exact(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{'id': str(i), 'question': '?', 'context': 'Exact\ncontext', 'answer': 'A',
                     'context_sha256': hashlib.sha256(b'Exact\ncontext').hexdigest(),
                     'rater_status': 'ok' if i < 2 else 'invalid',
                     'sufficiency_label': i if i < 2 else None} for i in range(3)]
            source = root / 'source.jsonl'
            write_rows(source, rows)
            args = SimpleNamespace(source=source, dataset_dir=root / 'data', limit=None)
            prepare(args)
            prepare(args)
            self.assertEqual(len(dataset(args)), 2)
            self.assertEqual(dataset(args)[0]['context'], 'Exact\ncontext')
            self.assertEqual(read_json(args.dataset_dir / 'manifest.json')['excluded_count'], 1)
            rows[2].update(rater_status='ok', sufficiency_label=1)
            write_rows(source, rows)
            with self.assertRaises(ValueError):
                prepare(args)

    def test_summary_uses_within_group_denominators_and_rejects_stale_judgment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.jsonl'
            import hashlib
            rows = [{'id': str(i), 'question': '?', 'context': 'C', 'answer': 'A',
                     'context_sha256': hashlib.sha256(b'C').hexdigest(), 'rater_status': 'ok',
                     'sufficiency_label': int(i < 2)} for i in range(4)]
            write_rows(source, rows)
            args = SimpleNamespace(source=source, dataset_dir=root / 'data', results_dir=root / 'results', limit=None, seed=42)
            prepare(args)
            folder = args.results_dir / 'llama31'
            responses = [{'id': str(i), 'context_sha256': rows[i]['context_sha256'], 'status': 'ok', 'raw_output': 'A'} for i in range(4)]
            write_rows(folder / 'responses.jsonl', responses)
            judgments = [{'id': str(i), 'status': 'ok', 'response_digest': response_digest(responses[i]),
                          'category': c, 'method': 'llm'} for i, c in enumerate(['correct', 'abstain', 'hallucinate', 'correct'])]
            judgments[-1]['response_digest'] = 'stale'
            write_rows(folder / 'judgments.jsonl', judgments)
            report(args)
            summary = {r['context_group']: r for r in read_json(args.results_dir / 'summary.json')}
            self.assertEqual(summary['sufficient']['correct_rate'], .5)
            self.assertEqual(summary['insufficient']['hallucinate_rate'], 1)
            self.assertEqual(summary['all']['unclassified'], 1)
            self.assertFalse(summary['all']['complete'])
            self.assertIsNone(read_rows(folder / 'categorized.jsonl')[-1]['category'])

    def test_configuration_change_cannot_silently_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            freeze(path, {'revision': 'a'})
            with self.assertRaises(ValueError):
                freeze(path, {'revision': 'b'})


if __name__ == '__main__':
    unittest.main()
