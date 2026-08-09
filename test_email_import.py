import email
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from emailExtraction import (
    anonymize_imported_email,
    extract_emails,
    message_to_email_data,
    save_imported_emails,
)
from preprocessing.anonymization import EmailAnonymizer


class TestEmailImportAnonymization(unittest.TestCase):
    def test_message_is_anonymized_before_it_is_saved(self):
        raw_message = (
            b"From: Ana Silva <ana.silva@example.com>\r\n"
            b"To: Joao Costa <joao.costa@example.com>\r\n"
            b"Subject: Reuniao com Ana\r\n"
            b"Date: Fri, 24 Jul 2026 10:30:00 +0100\r\n"
            b"Content-Type: text/plain; charset=utf-8\r\n"
            b"\r\n"
            b"Ola Ana, liga-me para 912345678.\r\n"
        )
        message = email.message_from_bytes(raw_message)
        imported = message_to_email_data(message, email_id="42")
        anonymized = anonymize_imported_email(
            imported,
            EmailAnonymizer(use_spacy=False),
        )

        serialized = json.dumps(anonymized, ensure_ascii=False).lower()
        self.assertNotIn("ana.silva@example.com", serialized)
        self.assertNotIn("joao.costa@example.com", serialized)
        self.assertNotIn("912345678", serialized)
        self.assertNotIn("\"original\"", serialized)
        self.assertIn("[email_", serialized)
        self.assertIn("[telefone_", serialized)
        self.assertEqual(
            anonymized["sent_datetime"],
            "2026-07-24T10:30:00+01:00",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "emails.json"
            save_imported_emails([anonymized], output_path)
            saved = output_path.read_text(encoding="utf-8").lower()
            self.assertNotIn("ana.silva@example.com", saved)
            self.assertNotIn("912345678", saved)

    def test_importer_processes_every_message(self):
        messages = {
            b"1": self._raw_message("Ana", "ana@example.com", "912345678"),
            b"2": self._raw_message("Joao", "joao@example.com", "913456789"),
        }

        class FakeIMAP:
            def __init__(self, server):
                self.server = server

            def login(self, account, password):
                return "OK", []

            def select(self, mailbox):
                return "OK", []

            def search(self, charset, criteria):
                return "OK", [b"1 2"]

            def fetch(self, email_id, query):
                return "OK", [(b"RFC822", messages[email_id])]

            def close(self):
                return "OK", []

            def logout(self):
                return "BYE", []

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "imported.json"
            environment = {
                "EMAIL": "account@example.com",
                "PASSWORD": "secret",
                "SERVER": "imap.example.com",
            }
            with patch.dict("os.environ", environment, clear=False), patch(
                "emailExtraction.imaplib.IMAP4_SSL",
                FakeIMAP,
            ):
                imported = extract_emails(
                    output_path=output_path,
                    use_spacy=False,
                )

            self.assertEqual(len(imported), 2)
            saved = output_path.read_text(encoding="utf-8").lower()
            self.assertNotIn("ana@example.com", saved)
            self.assertNotIn("joao@example.com", saved)
            self.assertNotIn("912345678", saved)
            self.assertNotIn("913456789", saved)

    @staticmethod
    def _raw_message(name, address, phone):
        return (
            f"From: {name} <{address}>\r\n"
            "To: Secretaria <secretaria@example.com>\r\n"
            f"Subject: Mensagem de {name}\r\n"
            "Date: Fri, 24 Jul 2026 10:30:00 +0100\r\n"
            "Content-Type: text/plain; charset=utf-8\r\n"
            "\r\n"
            f"Contacto: {phone}.\r\n"
        ).encode("utf-8")


if __name__ == "__main__":
    unittest.main()
