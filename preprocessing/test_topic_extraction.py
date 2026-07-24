import unittest

from dataset.generate_gold_from_metadata import build_topic
from preprocessing.argument_extraction import TopicExtractor


class TestTopicExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = TopicExtractor(nlp_model=None)

    def extract_one(self, body, subject="", intent="agendamento_reuniao"):
        results = self.extractor.extract(body, subject, intent)
        return results[0].text if results else None

    def test_extracts_topic_after_sobre(self):
        topic = self.extract_one(
            "Podemos remarcar a reuniao sobre planeamento de aulas "
            "para quando for conveniente?"
        )

        self.assertEqual(topic, "planeamento de aulas")

    def test_extracts_topic_after_discutir(self):
        topic = self.extract_one(
            "O objetivo e discutir credibilidade dos dados recolhidos."
        )

        self.assertEqual(topic, "credibilidade dos dados recolhidos")

    def test_extracts_point_of_situation_topic(self):
        topic = self.extract_one(
            "Vamos fechar o ponto de situacao de acompanhamento pedagogico."
        )

        self.assertEqual(topic, "acompanhamento pedagogico")

    def test_extracts_relative_documents_topic(self):
        topic = self.extract_one(
            "Confirmado para amanha. Levarei os documentos relativos "
            "a orientacao da dissertacao."
        )

        self.assertEqual(topic, "orientacao da dissertacao")

    def test_uses_informative_subject_when_body_omits_topic(self):
        topic = self.extract_one(
            "Surgiu uma sobreposicao no horario e nao poderei participar.",
            subject="RE: Imprevisto - revisao do plano de trabalhos",
            intent="cancelamento_reuniao",
        )

        self.assertEqual(topic, "revisao do plano de trabalhos")

    def test_generic_subject_and_body_return_empty(self):
        topic = self.extract_one(
            "Fica entao combinado para quinta as 16h.",
            subject="Fw: Confirmado",
            intent="reuniao_confirmada",
        )

        self.assertIsNone(topic)

    def test_non_meeting_does_not_emit_topic(self):
        topic = self.extract_one(
            "Envio o documento atualizado relativo a avaliacao continua.",
            subject="Atualizacao de processo",
            intent="nao_reuniao",
        )

        self.assertIsNone(topic)

    def test_returns_only_one_topic(self):
        results = self.extractor.extract(
            "Confirmo a reuniao para discutirmos horarios do semestre.",
            "Confirmacao - horarios do semestre",
            "reuniao_confirmada",
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "horarios do semestre")

    def test_extracts_topics_from_varied_explicit_anchors(self):
        cases = [
            (
                "A reuniao sera dedicada a validacao das atas.",
                "validacao das atas",
            ),
            (
                "Na reuniao vamos tratar da coordenacao do curso.",
                "coordenacao do curso",
            ),
            (
                "Proponho debater os criterios de avaliacao.",
                "criterios de avaliacao",
            ),
            (
                "A sessao incidira sobre a atualizacao do regulamento.",
                "atualizacao do regulamento",
            ),
        ]

        for body, expected in cases:
            with self.subTest(body=body):
                self.assertEqual(self.extract_one(body), expected)

    def test_rejects_vague_pronoun_as_topic(self):
        topic = self.extract_one(
            "Depois falamos sobre isso. Confirme apenas a disponibilidade."
        )

        self.assertIsNone(topic)

    def test_extracts_topic_from_colon_subject(self):
        topic = self.extract_one(
            "Podemos reunir amanha?",
            subject="Reuniao: acompanhamento pedagogico",
        )

        self.assertEqual(topic, "acompanhamento pedagogico")

    def test_gold_excludes_topic_missing_from_email(self):
        topic = build_topic(
            {
                "subject": "Reuniao confirmada",
                "body": "Confirmo a minha presenca.",
                "topic": "avaliacao continua",
            }
        )

        self.assertEqual(topic, [])

    def test_gold_accepts_topic_from_subject(self):
        topic = build_topic(
            {
                "subject": "Imprevisto - avaliacao continua",
                "body": "Nao poderei participar.",
                "topic": "avaliacao continua",
            }
        )

        self.assertEqual(topic, ["avaliacao continua"])


if __name__ == "__main__":
    unittest.main()
