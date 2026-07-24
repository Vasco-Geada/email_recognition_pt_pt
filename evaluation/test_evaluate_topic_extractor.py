import unittest
from types import SimpleNamespace

from evaluate_topic_extractor import evaluate_cases, normalize_text


class StubExtractor:
    def __init__(self, predictions):
        self.predictions = predictions

    def extract(self, text, subject="", intent=""):
        prediction = self.predictions.get(subject, "")
        if not prediction:
            return []
        return [
            SimpleNamespace(
                text=prediction,
                confidence=0.75,
                extraction_method="stub",
            )
        ]


class TestTopicEvaluator(unittest.TestCase):
    def test_normalization_ignores_case_accents_and_punctuation(self):
        self.assertEqual(
            normalize_text("  Avaliação: Contínua! "),
            "avaliacao continua",
        )

    def test_positive_and_negative_metrics_are_kept_separate(self):
        cases = [
            self._case("p1", "Plano Anual"),
            self._case("p2", "criterios de avaliacao"),
            self._case("n1", ""),
            self._case("n2", ""),
        ]
        extractor = StubExtractor(
            {
                "p1": "plano anual",
                "p2": "avaliacao",
                "n2": "horarios",
            }
        )

        metrics, rows = evaluate_cases(cases, extractor=extractor)

        self.assertEqual(len(rows), 4)
        self.assertAlmostEqual(metrics["overall_exact_match"], 0.5)
        self.assertAlmostEqual(metrics["positive_exact_match"], 0.5)
        self.assertAlmostEqual(metrics["positive_token_f1"], 0.75)
        self.assertAlmostEqual(metrics["negative_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["false_positive_rate"], 0.5)
        self.assertAlmostEqual(metrics["topic_detection_f1"], 0.8)

    @staticmethod
    def _case(case_id, gold_topic):
        return {
            "id": case_id,
            "split": "test",
            "subject": case_id,
            "body": "Corpo do email.",
            "label": "agendamento_reuniao",
            "gold_topic": gold_topic,
        }


if __name__ == "__main__":
    unittest.main()
