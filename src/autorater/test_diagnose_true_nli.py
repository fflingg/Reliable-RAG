"""Check actual reported SentencePiece label shapes without loading the 11B model."""
import unittest
from diagnose_true_nli import label_sequences, teacher_forcing_tokens


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return {'0': [3, 632], '1': [209]}[text]


class LabelSequenceTests(unittest.TestCase):
    def test_accepts_multi_token_zero(self):
        self.assertEqual(label_sequences(FakeTokenizer()), {'0': [3, 632], '1': [209]})

    def test_shift_preserves_all_label_tokens_and_scores_eos(self):
        self.assertEqual(teacher_forcing_tokens([3, 632], eos_id=1, decoder_start_id=0),
                         ([0, 3, 632], [3, 632, 1]))
        self.assertEqual(teacher_forcing_tokens([209], eos_id=1, decoder_start_id=0),
                         ([0, 209], [209, 1]))


if __name__ == '__main__':
    unittest.main()
