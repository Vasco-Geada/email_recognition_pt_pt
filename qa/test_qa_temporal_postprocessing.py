import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QA_DIR = PROJECT_ROOT / "qa"
for path in (str(PROJECT_ROOT), str(QA_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from qa_pipeline import QAContextEnricher


class TestQATemporalPostProcessing(unittest.TestCase):
    def setUp(self):
        self.enricher = QAContextEnricher()

    def test_full_day_and_time_expression_is_kept(self):
        enrichment = self.enricher.build_context(
            email_text="A reuniao fica para dia 18 as 9h30.",
            subject="Reuniao",
            metadata={
                "label": "agendamento_reuniao",
                "sent_datetime": "2026-03-23T11:00:00",
            },
        )

        self.assertEqual(len(enrichment["temporal_hints"]), 1)
        hint = enrichment["temporal_hints"][0]
        self.assertEqual(hint["text"], "dia 18 as 9h30")
        self.assertEqual(hint["canonical_value"], "2026-04-18T09:30:00")
        self.assertNotIn("2026-04-18", enrichment["context"])

    def test_partial_qa_answer_is_aligned_to_full_expression(self):
        email_text = "A reuniao fica para dia 18 as 9h30."
        enrichment = self.enricher.build_context(
            email_text=email_text,
            subject="Reuniao",
            metadata={
                "label": "agendamento_reuniao",
                "sent_datetime": "2026-03-23T11:00:00",
            },
        )
        results = self.enricher.apply_fallbacks(
            {
                "time": {
                    "answer": "9h30",
                    "confidence": 0.99,
                    "question": "Quando e a reuniao?",
                    "valid": True,
                },
                "participants": {},
                "location": {},
                "topic": {},
            },
            email_text=email_text,
            enrichment=enrichment,
        )

        self.assertEqual(results["time"]["answer"], "dia 18 as 9h30")
        self.assertEqual(
            results["time_normalized"]["answer"],
            "2026-04-18T09:30:00",
        )

    def test_non_meeting_temporal_phrase_is_not_a_meeting_time(self):
        email_text = "A bibliografia desta semana esta no Moodle."
        enrichment = self.enricher.build_context(
            email_text=email_text,
            subject="Bibliografia",
            metadata={
                "label": "nao_reuniao",
                "sent_datetime": "2026-03-08T12:30:00",
            },
        )
        results = self.enricher.apply_fallbacks(
            {
                "time": {
                    "answer": "esta semana",
                    "confidence": 0.95,
                    "valid": True,
                },
                "participants": {},
                "location": {},
                "topic": {},
            },
            email_text=email_text,
            enrichment=enrichment,
        )

        self.assertIsNone(results["time"]["answer"])
        self.assertIsNone(results["time_normalized"]["answer"])

    def test_meeting_without_explicit_time_rejects_hallucination(self):
        email_text = "Confirmo a reuniao. Enviarei a documentacao."
        enrichment = self.enricher.build_context(
            email_text=email_text,
            subject="Confirmacao",
            metadata={
                "label": "reuniao_confirmada",
                "sent_datetime": "2026-04-23T16:00:00",
            },
        )
        results = self.enricher.apply_fallbacks(
            {
                "time": {
                    "answer": "Secretaria Academica",
                    "confidence": 0.8,
                    "valid": True,
                },
                "participants": {},
                "location": {},
                "topic": {},
            },
            email_text=email_text,
            enrichment=enrichment,
        )

        self.assertIsNone(results["time"]["answer"])
        self.assertEqual(
            results["time"]["fallback_source"],
            "no_explicit_temporal_expression",
        )

    def test_time_of_day_is_an_interval_not_an_invented_hour(self):
        enrichment = self.enricher.build_context(
            email_text="Podemos reunir quarta a tarde?",
            subject="Reuniao",
            metadata={
                "label": "agendamento_reuniao",
                "sent_datetime": "2026-03-13T16:05:00",
            },
        )

        hint = enrichment["temporal_hints"][0]
        self.assertEqual(hint["temporal_type"], "interval")
        self.assertIsNone(hint["normalized_datetime"])
        self.assertEqual(
            hint["canonical_value"],
            "2026-03-18T12:00:00/2026-03-18T18:00:00",
        )

    def test_rfc_2822_email_date_is_supported(self):
        enrichment = self.enricher.build_context(
            email_text="Podemos reunir amanha as 10h?",
            subject="Reuniao",
            metadata={
                "label": "agendamento_reuniao",
                "Date": "Mon, 23 Mar 2026 11:00:00 +0000",
            },
        )

        hint = enrichment["temporal_hints"][0]
        self.assertEqual(hint["canonical_value"], "2026-03-24T10:00:00+00:00")

    def test_multiple_expressions_require_qa_disambiguation(self):
        email_text = "Podemos passar a reuniao de segunda as 10h para terca as 15h?"
        enrichment = self.enricher.build_context(
            email_text=email_text,
            subject="Remarcacao",
            metadata={
                "label": "agendamento_reuniao",
                "sent_datetime": "2026-03-20T11:00:00",
            },
        )
        results = self.enricher.apply_fallbacks(
            {
                "time": {
                    "answer": "texto incorreto",
                    "confidence": 0.9,
                    "valid": True,
                },
                "participants": {},
                "location": {},
                "topic": {},
            },
            email_text=email_text,
            enrichment=enrichment,
        )

        self.assertIsNone(results["time"]["answer"])
        self.assertEqual(
            results["time"]["fallback_source"],
            "ambiguous_temporal_expressions",
        )


if __name__ == "__main__":
    unittest.main()
