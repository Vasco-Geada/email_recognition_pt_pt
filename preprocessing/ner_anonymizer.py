# -*- coding: utf-8 -*-
"""
spaCy NER and academic-context heuristics for anonymization.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from preprocessing.anonymization_config import (
    ACADEMIC_TITLES,
    COMMON_PT_NAMES,
    DEFAULT_SPACY_MODEL,
    EntityCandidate,
    PLATFORM_WHITELIST,
)


logger = logging.getLogger(__name__)


class NERAnonymizer:
    """
    Detect person, organization and location names using spaCy Portuguese NER.

    If spaCy or the configured model is unavailable, deterministic heuristics
    still cover common informal academic patterns.
    """

    SPACY_TYPE_MAP = {
        "PER": "PERSON",
        "PERSON": "PERSON",
        "ORG": "ORG",
        "LOC": "LOCAL",
        "GPE": "LOCAL",
    }

    def __init__(self, model_name: str = DEFAULT_SPACY_MODEL, use_spacy: bool = True) -> None:
        self.model_name = model_name
        self.nlp = None

        if use_spacy:
            try:
                import spacy

                self.nlp = spacy.load(model_name)
                logger.info("Loaded spaCy model for anonymization: %s", model_name)
            except Exception as exc:  # pragma: no cover - environment-dependent fallback
                logger.warning("spaCy model unavailable (%s). Using heuristics only.", exc)
                self.nlp = None

        title_pattern = "|".join(re.escape(title) for title in ACADEMIC_TITLES)
        self.title_name_pattern = re.compile(
            rf"\b(?i:(?:{title_pattern}))\s+([A-ZÁÉÍÓÚÀÂÊÕÇ][\wÀ-ÿ-]+"
            rf"(?:\s+[A-ZÁÉÍÓÚÀÂÊÕÇ][\wÀ-ÿ-]+){{0,3}})"
        )
        self.greeting_name_pattern = re.compile(
            r"\b(?i:(?:olá|ola|boas|car[oa]|bom dia|boa tarde))\s+"
            r"([A-ZÁÉÍÓÚÀÂÊÕÇ][\wÀ-ÿ-]+(?:\s+[A-ZÁÉÍÓÚÀÂÊÕÇ][\wÀ-ÿ-]+){0,2})"
        )

    def find(self, text: str) -> List[EntityCandidate]:
        """Return NER and heuristic candidates."""
        candidates: List[EntityCandidate] = []
        candidates.extend(self._find_spacy_entities(text))
        candidates.extend(self._find_academic_person_patterns(text))
        candidates.extend(self._find_common_names(text))
        return candidates

    def _find_spacy_entities(self, text: str) -> List[EntityCandidate]:
        if self.nlp is None or not text.strip():
            return []

        candidates: List[EntityCandidate] = []
        doc = self.nlp(text)
        for ent in doc.ents:
            entity_type = self.SPACY_TYPE_MAP.get(ent.label_)
            if not entity_type:
                continue
            if self._is_whitelisted(ent.text):
                continue
            candidates.append(
                EntityCandidate(
                    ent.text,
                    entity_type,
                    ent.start_char,
                    ent.end_char,
                    "NER",
                    70,
                )
            )
        return candidates

    def _find_academic_person_patterns(self, text: str) -> List[EntityCandidate]:
        candidates: List[EntityCandidate] = []
        for pattern in (self.title_name_pattern, self.greeting_name_pattern):
            for match in pattern.finditer(text):
                original = match.group(1).strip()
                if self._is_whitelisted(original):
                    continue
                start = match.start(1)
                candidates.append(
                    EntityCandidate(
                        original,
                        "PERSON",
                        start,
                        start + len(original),
                        "ACADEMIC_RULE",
                        80,
                    )
                )
        return candidates

    def _find_common_names(self, text: str) -> List[EntityCandidate]:
        candidates: List[EntityCandidate] = []
        for name in COMMON_PT_NAMES:
            pattern = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE)
            for match in pattern.finditer(text):
                original = match.group()
                if self._is_whitelisted(original):
                    continue
                candidates.append(
                    EntityCandidate(
                        original,
                        "PERSON",
                        match.start(),
                        match.end(),
                        "NAME_LEXICON",
                        60,
                    )
                )
        return candidates

    @staticmethod
    def _is_whitelisted(text: str) -> bool:
        return text.strip().lower() in PLATFORM_WHITELIST
