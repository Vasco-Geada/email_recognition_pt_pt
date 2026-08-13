# -*- coding: utf-8 -*-
"""Import emails from IMAP and persist only their anonymized representation."""

from __future__ import annotations

import argparse
import email
import html
import imaplib
import json
import os
import re
from datetime import datetime, timezone
from email.header import decode_header
from email.message import Message
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

from preprocessing.anonymization import EmailAnonymizer


DEFAULT_OUTPUT = Path("dataset/imported_emails_anonymized.json")


def decode_str(value: Optional[str]) -> str:
    """Decode every MIME header fragment without exposing undecoded bytes."""
    if not value:
        return ""

    fragments = []
    for decoded, charset in decode_header(value):
        if isinstance(decoded, bytes):
            fragments.append(
                decoded.decode(charset or "utf-8", errors="replace")
            )
        else:
            fragments.append(str(decoded))
    return "".join(fragments).strip()


def _decode_payload(part: Message) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        raw_payload = part.get_payload()
        return raw_payload if isinstance(raw_payload, str) else ""
    return payload.decode(part.get_content_charset() or "utf-8", errors="replace")


def _html_to_text(value: str) -> str:
    without_blocks = re.sub(
        r"(?is)<(?:script|style).*?>.*?</(?:script|style)>",
        " ",
        value,
    )
    without_tags = re.sub(r"(?s)<[^>]+>", " ", without_blocks)
    return re.sub(r"\s+", " ", html.unescape(without_tags)).strip()


def get_email_body(message: Message) -> str:
    """Prefer plain text and use a stripped HTML body only as fallback."""
    html_fallback = ""
    if message.is_multipart():
        for part in message.walk():
            disposition = str(part.get("Content-Disposition", "")).lower()
            if "attachment" in disposition:
                continue
            content_type = part.get_content_type()
            if content_type == "text/plain":
                return _decode_payload(part).strip()
            if content_type == "text/html" and not html_fallback:
                html_fallback = _html_to_text(_decode_payload(part))
        return html_fallback

    content = _decode_payload(message)
    if message.get_content_type() == "text/html":
        return _html_to_text(content)
    return content.strip()


def _sent_datetime(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat()


def message_to_email_data(message: Message, email_id: str) -> Dict:
    """Convert one parsed message into the project's raw email structure."""
    date_header = message.get("Date")
    result = {
        "email_id": str(email_id),
        "email_date": date_header,
        "sent_datetime": _sent_datetime(date_header),
        "subject": decode_str(message.get("Subject")),
        "body": get_email_body(message),
        "sender": decode_str(message.get("From")),
        "recipient": decode_str(message.get("To")),
    }
    cc = decode_str(message.get("Cc"))
    if cc:
        result["cc"] = cc
    return result

# Anonymization is done immediately after fetching the email to avoid storing any sensitive data in memory or on disk. 
# The anonymized representation is then saved to a JSON file for downstream processing.
def anonymize_imported_email(
    email_data: Dict,
    anonymizer: EmailAnonymizer,
) -> Dict:
    """Anonymize before the email is returned to any downstream component."""
    anonymized = anonymizer.anonymize_email(
        email_data,
        keep_mapping=False,
        include_original_text=False,
    )
    anonymized["imported_at"] = datetime.now(timezone.utc).isoformat()
    return anonymized


def save_imported_emails(emails: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(emails, handle, ensure_ascii=False, indent=2)

# Fetch emails from an IMAP server, anonymize them immediately, and save the anonymized dataset to a JSON file.
def extract_emails(
    output_path: Path = DEFAULT_OUTPUT,
    mailbox: str = "inbox",
    search_criteria: str = "ALL",
    limit: Optional[int] = None,
    use_spacy: bool = True,
) -> List[Dict]:
    """Fetch IMAP messages, anonymize immediately and save a JSON dataset."""
    load_dotenv()
    account = os.getenv("EMAIL")
    password = os.getenv("PASSWORD")
    server = os.getenv("SERVER")
    missing = [
        name
        for name, value in {
            "EMAIL": account,
            "PASSWORD": password,
            "SERVER": server,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(
            "Variaveis em falta no .env: " + ", ".join(missing)
        )

    anonymizer = EmailAnonymizer(use_spacy=use_spacy)
    imported: List[Dict] = []
    mail = imaplib.IMAP4_SSL(server)
    try:
        mail.login(account, password)
        status, _ = mail.select(mailbox)
        if status != "OK":
            raise RuntimeError(f"Nao foi possivel abrir a mailbox: {mailbox}")

        status, messages = mail.search(None, search_criteria)
        if status != "OK":
            raise RuntimeError(
                f"Pesquisa IMAP falhou: {search_criteria}"
            )
        email_ids = messages[0].split()
        if limit is not None:
            email_ids = email_ids[-limit:]

        for raw_id in email_ids:
            status, message_data = mail.fetch(raw_id, "(RFC822)")
            if status != "OK":
                continue
            raw_message = next(
                (
                    item[1]
                    for item in message_data
                    if isinstance(item, tuple)
                    and len(item) > 1
                    and isinstance(item[1], bytes)
                ),
                None,
            )
            if raw_message is None:
                continue
            parsed = email.message_from_bytes(raw_message)
            raw_data = message_to_email_data(
                parsed,
                email_id=raw_id.decode(errors="replace"),
            )
            imported.append(
                anonymize_imported_email(raw_data, anonymizer)
            )
    finally:
        try:
            mail.close()
        except (imaplib.IMAP4.error, OSError):
            pass
        try:
            mail.logout()
        except (imaplib.IMAP4.error, OSError):
            pass

    save_imported_emails(imported, output_path)
    print(f"{len(imported)} emails anonimizados guardados em: {output_path.resolve()}")
    return imported

# Getter of the extracted emails.
def GetExtractEmail() -> List[Dict]:
    """Compatibility wrapper for the original project entry point."""
    return extract_emails()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Importa emails via IMAP e grava apenas a versao anonimizada."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mailbox", default="inbox")
    parser.add_argument("--search", default="ALL")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--disable-spacy",
        action="store_true",
        help="Usa apenas regras/regex na anonimizacao.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_emails(
        output_path=args.output,
        mailbox=args.mailbox,
        search_criteria=args.search,
        limit=args.limit,
        use_spacy=not args.disable_spacy,
    )


if __name__ == "__main__":
    main()
