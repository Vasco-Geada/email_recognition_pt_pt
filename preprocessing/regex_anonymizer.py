# -*- coding: utf-8 -*-
"""
Regex-based detection of sensitive data in Portuguese academic emails.
"""

from __future__ import annotations

import re
from typing import List

from preprocessing.anonymization_config import EntityCandidate


class RegexAnonymizer:
    """Detect emails, phone numbers, URLs and academic identifiers."""

    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )
    URL_PATTERN = re.compile(
        r"\b(?:https?://|www\.)[^\s<>\"]+",
        re.IGNORECASE,
    )
    PHONE_PATTERN = re.compile(
        r"(?<!\w)(?:\+351\s*)?(?:9[1236]\d(?:[\s.-]?\d{3}){2}|2\d{2}(?:[\s.-]?\d{3}){2})(?!\w)"
    )
    STUDENT_ID_CONTEXT_PATTERN = re.compile(
        r"\b(?:aluno|aluna|estudante|n[úu]mero|n[ºo]\.?)\s*"
        r"(?:n[ºo]\.?\s*)?(?:de\s*)?(?:aluno\s*)?(?:[:#-]?\s*)"
        r"(\d{5,10})\b",
        re.IGNORECASE,
    )
    STUDENT_ID_EXPLICIT_PATTERN = re.compile(
        r"\b(?:a|aluno|ist|isel|fc|uc|up|ul)\d{5,10}\b",
        re.IGNORECASE,
    )

    def find(self, text: str) -> List[EntityCandidate]:
        """Return regex entity candidates for a text field."""
        candidates: List[EntityCandidate] = []

        for match in self.EMAIL_PATTERN.finditer(text):
            candidates.append(
                EntityCandidate(match.group(), "EMAIL", match.start(), match.end(), "REGEX", 100)
            )

        for match in self.URL_PATTERN.finditer(text):
            candidates.append(
                EntityCandidate(match.group(), "URL", match.start(), match.end(), "REGEX", 95)
            )

        for match in self.PHONE_PATTERN.finditer(text):
            candidates.append(
                EntityCandidate(match.group(), "PHONE", match.start(), match.end(), "REGEX", 90)
            )

        for match in self.STUDENT_ID_CONTEXT_PATTERN.finditer(text):
            start, end = match.span(1)
            candidates.append(
                EntityCandidate(match.group(1), "STUDENT_ID", start, end, "REGEX", 90)
            )

        for match in self.STUDENT_ID_EXPLICIT_PATTERN.finditer(text):
            candidates.append(
                EntityCandidate(
                    match.group(), "STUDENT_ID", match.start(), match.end(), "REGEX", 90
                )
            )

        return candidates

