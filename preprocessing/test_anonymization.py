# -*- coding: utf-8 -*-
"""
Tests for the email anonymization module.

Run:
    python preprocessing/test_anonymization.py
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.anonymization import EmailAnonymizer


def make_anonymizer() -> EmailAnonymizer:
    return EmailAnonymizer(use_spacy=True)


def anonymize_body(text: str) -> str:
    anonymizer = make_anonymizer()
    email = {"subject": "", "body": text, "label": "agendamento_reuniao"}
    return anonymizer.anonymize_email(email)["body"]


def test_simple_name() -> None:
    assert anonymize_body("Olá Ana, podemos reunir amanhã?") == (
        "Olá [PESSOA_1], podemos reunir amanhã?"
    )


def test_repeated_name() -> None:
    assert anonymize_body("Ana confirmou. Ana vai estar presente.") == (
        "[PESSOA_1] confirmou. [PESSOA_1] vai estar presente."
    )


def test_email() -> None:
    assert anonymize_body("Contacta-me em vasco@gmail.com") == (
        "Contacta-me em [EMAIL_1]"
    )


def test_phone() -> None:
    assert anonymize_body("O meu número é 912345678") == (
        "O meu número é [TELEFONE_1]"
    )


def test_university() -> None:
    assert anonymize_body("Sou aluno da Universidade de Lisboa") == (
        "Sou aluno da [UNIVERSIDADE_1]"
    )


def test_apps_are_not_anonymized() -> None:
    assert anonymize_body("Reunimos no Teams ou no Zoom") == (
        "Reunimos no Teams ou no Zoom"
    )


def test_mixed_case() -> None:
    assert anonymize_body(
        "Boas João, falamos no Teams? O professor Silva enviou email para joao@isel.pt"
    ) == (
        "Boas [PESSOA_1], falamos no Teams? "
        "O professor [PESSOA_2] enviou email para [EMAIL_1]"
    )


def test_subject_body_consistency() -> None:
    anonymizer = make_anonymizer()
    result = anonymizer.anonymize_email(
        {
            "subject": "Reunião com Ana",
            "body": "Boas Ana, confirmas no Teams?",
            "label": "agendamento_reuniao",
        }
    )
    assert result["subject"] == "Reunião com [PESSOA_1]"
    assert result["body"] == "Boas [PESSOA_1], confirmas no Teams?"


def run_all_tests() -> None:
    tests = [
        test_simple_name,
        test_repeated_name,
        test_email,
        test_phone,
        test_university,
        test_apps_are_not_anonymized,
        test_mixed_case,
        test_subject_body_consistency,
    ]
    for test in tests:
        test()
        print(f"OK - {test.__name__}")


if __name__ == "__main__":
    run_all_tests()

