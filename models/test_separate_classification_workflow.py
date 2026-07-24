import json
import tempfile
import unittest
from pathlib import Path

from run_classification_models import (
    MODEL_NAMES,
    evaluate_saved_models,
    file_sha256,
    load_all_models,
    model_artifact_paths,
    train_all_models,
)


class TestSeparateClassificationWorkflow(unittest.TestCase):
    def test_train_and_evaluate_use_separate_datasets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_path = root / "train.json"
            test_path = root / "test.json"
            model_dir = root / "models"
            output_dir = root / "results"
            self._write_dataset(train_path, self._training_emails())
            self._write_dataset(test_path, self._test_emails())

            metadata = train_all_models(
                dataset_path=train_path,
                model_dir=model_dir,
                use_anonymization=False,
                max_features=200,
                random_state=7,
            )

            self.assertEqual(metadata["num_training_examples"], 16)
            self.assertEqual(len(metadata["labels"]), 4)
            artifact_hashes = {}
            for model_name in MODEL_NAMES:
                model_path, vectorizer_path = model_artifact_paths(
                    model_dir,
                    model_name,
                )
                self.assertTrue(model_path.exists())
                self.assertTrue(vectorizer_path.exists())
                artifact_hashes[model_path] = file_sha256(model_path)
                artifact_hashes[vectorizer_path] = file_sha256(vectorizer_path)

            models = load_all_models(model_dir)
            vocabulary = set(
                models["logistic_regression"]
                .vectorizer
                .get_feature_names_out()
            )
            self.assertNotIn("palavraineditateste", vocabulary)

            summary = evaluate_saved_models(
                dataset_path=test_path,
                model_dir=model_dir,
                output_dir=output_dir,
            )

            self.assertEqual(summary["num_test"], 4)
            self.assertEqual(summary["overlapping_email_count"], 0)
            self.assertEqual(set(summary["metrics"]), set(MODEL_NAMES))
            self.assertTrue((output_dir / "summary.json").exists())
            for artifact_path, original_hash in artifact_hashes.items():
                self.assertEqual(file_sha256(artifact_path), original_hash)

            with self.assertRaisesRegex(ValueError, "mesmo usado no treino"):
                evaluate_saved_models(
                    dataset_path=train_path,
                    model_dir=model_dir,
                    output_dir=root / "invalid-results",
                )

    @staticmethod
    def _write_dataset(path, emails):
        with path.open("w", encoding="utf-8") as handle:
            json.dump(emails, handle, ensure_ascii=False, indent=2)

    @staticmethod
    def _training_emails():
        examples = {
            "agendamento_reuniao": [
                ("Marcar reunião", "Podemos agendar uma reunião para segunda?"),
                ("Disponibilidade", "Tem disponibilidade para reunirmos amanhã?"),
                ("Novo encontro", "Proponho marcar um encontro esta semana."),
                ("Remarcação", "Precisamos de remarcar a reunião para sexta."),
            ],
            "cancelamento_reuniao": [
                ("Cancelar reunião", "Tenho de cancelar a reunião de amanhã."),
                ("Impossibilidade", "Não poderei comparecer ao encontro marcado."),
                ("Reunião sem efeito", "A reunião fica cancelada por indisponibilidade."),
                ("Adiamento", "Peço o cancelamento da reunião desta tarde."),
            ],
            "reuniao_confirmada": [
                ("Confirmação", "Confirmo a minha presença na reunião."),
                ("Reunião confirmada", "Fica combinado para terça às dez."),
                ("Presença", "Estarei presente no encontro conforme acordado."),
                ("Tudo certo", "Confirmado para amanhã na sala indicada."),
            ],
            "nao_reuniao": [
                ("Documento", "Envio o relatório solicitado em anexo."),
                ("Informação", "A secretaria publicou os resultados finais."),
                ("Prazo", "O prazo de inscrição termina esta sexta-feira."),
                ("Aviso", "A plataforma académica estará indisponível hoje."),
            ],
        }
        emails = []
        for label, values in examples.items():
            for subject, body in values:
                emails.append({
                    "subject": subject,
                    "body": body,
                    "label": label,
                })
        return emails

    @staticmethod
    def _test_emails():
        return [
            {
                "subject": "Pedido de agenda",
                "body": "Conseguimos reunir na próxima quarta palavraineditateste?",
                "label": "agendamento_reuniao",
            },
            {
                "subject": "Desmarcação",
                "body": "Afinal não consigo estar na reunião agendada.",
                "label": "cancelamento_reuniao",
            },
            {
                "subject": "Aceite",
                "body": "Confirmo que estarei presente à hora combinada.",
                "label": "reuniao_confirmada",
            },
            {
                "subject": "Pauta disponível",
                "body": "A pauta das classificações já se encontra publicada.",
                "label": "nao_reuniao",
            },
        ]


if __name__ == "__main__":
    unittest.main()
