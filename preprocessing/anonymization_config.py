# -*- coding: utf-8 -*-
"""
Configuration and shared data structures for email anonymization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


DEFAULT_SPACY_MODEL = "pt_core_news_sm"


@dataclass(frozen=True)
class EntityCandidate:
    """Sensitive entity candidate found in a single text field."""

    original: str
    entity_type: str
    start: int
    end: int
    method: str
    priority: int = 50

    @property
    def length(self) -> int:
        return self.end - self.start


PLATFORM_WHITELIST = {
    "teams",
    "microsoft teams",
    "zoom",
    "discord",
    "moodle",
    "github",
    "google meet",
    "outlook",
    "gmail",
    "slack",
    "trello",
    "notion",
}


UNIVERSITIES = [
    "Universidade de Lisboa",
    "Universidade do Porto",
    "Universidade de Coimbra",
    "Universidade do Minho",
    "Universidade de Aveiro",
    "Universidade Nova de Lisboa",
    "NOVA",
]


INSTITUTIONS = [
    "Instituto Politécnico de Santarém",
    "Escola Superior de Gestão",
    "Instituto Superior Técnico",
    "Instituto Superior de Engenharia de Lisboa",
    "ISEL",
    "ISCTE",
]


ACADEMIC_TITLES = [
    "professor",
    "professora",
    "prof.",
    "profa.",
    "doutor",
    "doutora",
    "dr.",
    "dra.",
    "orientador",
    "orientadora",
    "aluno",
    "aluna",
]


# Conservative list for informal academic emails and tests. It is used only for
# exact word-boundary matches and after the platform whitelist is applied.
COMMON_PT_NAMES = {
    "ana",
    "joão",
    "joao",
    "maria",
    "rita",
    "paulo",
    "pedro",
    "vasco",
    "silva",
    "costa",
    "santos",
    "sofia",
    "miguel",
    "ines",
    "inês",
    "catarina",
    "diogo",
    "tiago",
    "beatriz",
}


PLACEHOLDER_PREFIX: Dict[str, str] = {
    "EMAIL": "EMAIL",
    "PHONE": "TELEFONE",
    "URL": "URL",
    "STUDENT_ID": "ID_ALUNO",
    "PERSON": "PESSOA",
    "ORG": "ORG",
    "LOCAL": "LOCAL",
    "UNIVERSITY": "UNIVERSIDADE",
    "INSTITUTION": "INSTITUICAO",
}

