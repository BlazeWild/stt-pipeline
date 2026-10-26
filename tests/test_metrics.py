import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from stt_metrics import char_edits, entity_score, normalize, word_edits


class MetricsTest(unittest.TestCase):
    def test_normalize_digits_punctuation_and_space(self):
        self.assertEqual(normalize("  वडा नं. १२!  "), "वडा नं 12")

    def test_word_edits(self):
        edits = word_edits("म घर जान्छु", "म जान्छु")
        self.assertEqual((edits.substitutions, edits.deletions, edits.insertions), (0, 1, 0))
        self.assertAlmostEqual(edits.error_rate, 1 / 3)

    def test_empty_reference_hallucination(self):
        self.assertEqual(char_edits("", "नमस्ते").error_rate, 1.0)

    def test_entity(self):
        exact, similarity = entity_score("वडा १२", "उहाँ वडा 12 मा बस्छन्")
        self.assertTrue(exact)
        self.assertEqual(similarity, 1.0)

    def test_entity_requires_word_boundaries(self):
        exact, _ = entity_score("राम", "आराम")
        self.assertFalse(exact)


if __name__ == "__main__":
    unittest.main()
