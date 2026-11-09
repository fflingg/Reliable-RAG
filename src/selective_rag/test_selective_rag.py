import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import selective_rag as sr


def fixture(root, n=90):
    rows = []
    for i in range(n):
        rows.append({'id': str(i), 'question': f'Question {i}?', 'context': 'Evidence',
                     'context_sha256': hashlib.sha256(b'Evidence').hexdigest(),
                     'answer': 'SECRET_REFERENCE', 'sufficiency_label': i % 2})
    data, behavior, out = root / 'data', root / 'behavior', root / 'out'
    sr.rb.write_rows(data / 'dataset.jsonl', rows)
    sr.rb.write_json(data / 'manifest.json', {'rows_digest': sr.rb.digest(rows)})
    folder = behavior / 'mistral'
    gen = {'dataset_digest': sr.rb.digest(rows), 'model': sr.rb.MODELS['mistral'],
           'revision': 'fixed-commit', 'instruction': sr.rb.GEN_INSTRUCTION, 'demo': sr.rb.DEMO}
    sr.rb.write_json(folder / 'generation_config.json', gen)
    sr.rb.write_json(folder / 'evaluation_config.json', {'dataset_digest': sr.rb.digest(rows)})
    responses = [{'id': r['id'], 'context_sha256': r['context_sha256'], 'status': 'ok',
                  'raw_output': 'The answer is: proposed'} for r in rows]
    judgments = [{'id': r['id'], 'response_digest': sr.rb.response_digest(r), 'status': 'ok',
                  'category': ['correct', 'hallucinate', 'abstain'][i % 3]} for i, r in enumerate(responses)]
    sr.rb.write_rows(folder / 'responses.jsonl', responses)
    sr.rb.write_rows(folder / 'judgments.jsonl', judgments)
    args = SimpleNamespace(dataset_dir=data, behavior_dir=behavior, output_dir=out,
                           model='mistral', seed=42, limit=None, allow_partial=False,
                           samples=2, self_evaluations=2, sample_max_new_tokens=512,
                           max_input_tokens=16384, searches=3, coverages=[0, .3, .6, 1])
    sr.split(args)
    return args


class SelectiveTests(unittest.TestCase):
    def test_question_groups_stay_together_and_split_does_not_use_outcomes(self):
        rows = [{'id': str(i), 'question': str(i)} for i in range(30)]
        rows.append({'id': 'dup', 'question': '  1  '})
        first = sr.make_split(rows, 42)
        for p in first['partitions'].values():
            self.assertEqual('1' in p, 'dup' in p)
        self.assertEqual(first['partitions'], sr.make_split(list(reversed(rows)), 42)['partitions'])

    def test_no_reference_label_or_rater_leaks_to_self_evaluation(self):
        row = {'question': 'Where?', 'context': 'Documents', 'answer': 'SECRET_REFERENCE',
               'sufficiency_label': 'SECRET_LABEL', 'category': 'SECRET_CATEGORY',
               'rater_raw_output': 'SECRET_RATER', 'response': {'raw_output': 'The answer is: Rome'}}
        prompt = json.dumps(sr.self_messages(row, [{'raw_output': 'The answer is: Paris'}], 42))
        self.assertNotIn('SECRET', prompt)
        self.assertIn('Rome', prompt)
        self.assertIn('Paris', prompt)

    def test_native_abstentions_and_zero_coverage_and_ties(self):
        rows = [{'category': c} for c in ['correct', 'hallucinate', 'abstain', 'correct']]
        self.assertEqual(sr.metrics(rows, [1, 1, 1, 0], 1)['coverage'], .5)
        self.assertEqual(sr.metrics(rows, [1, 1, 1, 0], 1)['selective_accuracy'], .5)
        self.assertIsNone(sr.metrics(rows, [1]*4, float('inf'))['selective_accuracy'])
        t = sr.choose_threshold(rows, [1, 1, 0, 1], .5)
        self.assertIn(sr.metrics(rows, [1, 1, 0, 1], t)['answered'], (0, 3))

    def test_partial_and_stale_runs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            path = args.behavior_dir / 'mistral/judgments.jsonl'
            rows = sr.rb.read_rows(path)
            rows[0]['response_digest'] = 'stale'
            sr.rb.write_rows(path, rows)
            with self.assertRaisesRegex(ValueError, 'unclassified'):
                sr.cohort(args)
            args.allow_partial = True
            usable, _, _ = sr.cohort(args)
            self.assertEqual(len(usable), 89)

    def test_resume_after_interruption_and_freeze(self):
        class Engine:
            sample_calls = 0
            rating_calls = 0
            fail = True
            def __init__(self, *a):
                pass
            def close(self):
                pass
            def sample(self, *a):
                Engine.sample_calls += 1
                if Engine.sample_calls == 2 and Engine.fail:
                    raise RuntimeError('interruption')
                return {'raw_output': 'The answer is: guess', 'truncated': False}
            def probability(self, *a):
                Engine.rating_calls += 1
                return {'p_true': .7, 'label_probability_mass': .8}
        with tempfile.TemporaryDirectory() as tmp:
            args = fixture(Path(tmp))
            args.limit = 1
            with patch.object(sr, 'ConfidenceEngine', Engine), patch.object(sr.rb, 'environment', return_value={}):
                with self.assertRaisesRegex(RuntimeError, 'interruption'):
                    sr.confidence(args)
                Engine.fail = False
                sr.confidence(args)
                self.assertEqual(Engine.sample_calls, 3)  # One persisted sample reused.
                self.assertEqual(Engine.rating_calls, 2)
                sr.confidence(args)
                self.assertEqual(Engine.rating_calls, 2)
                args.samples = 3
                with self.assertRaisesRegex(ValueError, 'Frozen configuration'):
                    sr.confidence(args)

    def test_cpu_end_to_end_and_test_labels_do_not_affect_model_or_thresholds(self):
        try:
            import sklearn
        except ImportError:
            self.skipTest('Install requirements.txt to test CPU training')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = fixture(root)
            rows, gen, partition = sr.cohort(args)
            folder = args.output_dir / args.model
            sr.rb.write_json(folder / 'confidence_config.json', {'aggregation': 'SYNTHETIC TEST ONLY'})
            sr.rb.write_rows(folder / 'confidence.jsonl', [
                {'id': r['id'], 'response_digest': r['response_digest'], 'status': 'ok',
                 'confidence': None if r['category'] == 'abstain' else (.8 if int(r['id']) % 2 else .2)}
                for r in rows])
            sr.fit(args)
            first = sr.rb.read_json(folder / 'combined_lr.json')
            with (folder / 'operating_points.csv').open(encoding='utf-8-sig') as stream:
                import csv
                thresholds = [r['threshold'] for r in csv.DictReader(stream)]
            # Flip only held-out answer outcomes in a fresh output tree.
            path = args.behavior_dir / args.model / 'judgments.jsonl'
            judgments = sr.rb.read_rows(path)
            test_ids = set(partition['partitions']['test'])
            for r in judgments:
                if r['id'] in test_ids and r['category'] != 'abstain':
                    r['category'] = 'correct' if r['category'] == 'hallucinate' else 'hallucinate'
            sr.rb.write_rows(path, judgments)
            import shutil
            new_out = root / 'second'
            shutil.copytree(args.output_dir, new_out)
            for f in ('cohort.json', 'fit_config.json'):
                (new_out / args.model / f).unlink()
            args.output_dir = new_out
            sr.fit(args)
            self.assertEqual(first, sr.rb.read_json(new_out / args.model / 'combined_lr.json'))
            with (new_out / args.model / 'operating_points.csv').open(encoding='utf-8-sig') as stream:
                self.assertEqual(thresholds, [r['threshold'] for r in csv.DictReader(stream)])
            sr.plot(args)
            self.assertTrue((new_out / args.model / 'accuracy_coverage.png').exists())


if __name__ == '__main__':
    unittest.main()
