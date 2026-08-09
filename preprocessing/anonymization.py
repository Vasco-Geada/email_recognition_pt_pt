# -*- coding: utf-8 -*-
"""
Hybrid anonymization for Portuguese academic emails.

Recommended pipeline:
    raw email -> anonymization -> preprocessing -> intent classification
    -> trigger extraction -> argument extraction -> temporal normalization
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.anonymization_config import (
    INSTITUTIONS,
    PLATFORM_WHITELIST,
    PLACEHOLDER_PREFIX,
    UNIVERSITIES,
    EntityCandidate,
)
from preprocessing.ner_anonymizer import NERAnonymizer
from preprocessing.regex_anonymizer import RegexAnonymizer


logger = logging.getLogger(__name__)


@dataclass
class AnonymizedEntity:
    """Entity replacement metadata for one field."""

    original: Optional[str]
    replacement: str
    type: str
    start: int
    end: int
    method: str
    field: str

    def to_dict(self, include_original: bool = False) -> Dict:
        data = asdict(self)
        if not include_original:
            data.pop("original", None)
        return data


class PlaceholderRegistry:
    """Keeps placeholders consistent within one email."""

    def __init__(self) -> None:
        self._mapping: Dict[Tuple[str, str], str] = {}
        self._counters: Dict[str, int] = {}

    def get(self, original: str, entity_type: str) -> str:
        key = (entity_type, self._normalize_key(original))
        if key in self._mapping:
            return self._mapping[key]

        prefix = PLACEHOLDER_PREFIX.get(entity_type, entity_type)
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        placeholder = f"[{prefix}_{self._counters[prefix]}]"
        self._mapping[key] = placeholder
        return placeholder

    def export_mapping(self) -> Dict[str, str]:
        """Return a pseudonymization mapping for controlled review."""
        return {
            f"{entity_type}:{original_key}": replacement
            for (entity_type, original_key), replacement in self._mapping.items()
        }

    @staticmethod
    def _normalize_key(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower())


class EmailAnonymizer:
    """Main hybrid anonymizer for email text and participant metadata."""

    SENSITIVE_METADATA_FIELDS = (
        "sender",
        "from",
        "from_",
        "recipient",
        "recipients",
        "to",
        "cc",
        "bcc",
        "participants",
    )

    def __init__(self, use_spacy: bool = True) -> None:
        self.regex_anonymizer = RegexAnonymizer()
        self.ner_anonymizer = NERAnonymizer(use_spacy=use_spacy)
        self.academic_patterns = self._compile_academic_patterns()

    def anonymize_text(
        self,
        text: str,
        field: str = "text",
        registry: Optional[PlaceholderRegistry] = None,
        include_original: bool = False,
    ) -> Tuple[str, List[Dict]]:
        """Anonymize one text field and return text plus entity metadata."""
        if registry is None:
            registry = PlaceholderRegistry()

        safe_text = "" if text is None else str(text)
        candidates = self._find_candidates(safe_text)
        selected = self._resolve_overlaps(candidates)

        replacements: List[Tuple[EntityCandidate, str]] = []
        entities: List[AnonymizedEntity] = []
        for candidate in selected:
            replacement = registry.get(candidate.original, candidate.entity_type)
            replacements.append((candidate, replacement))
            entities.append(
                AnonymizedEntity(
                    original=candidate.original if include_original else None,
                    replacement=replacement,
                    type=candidate.entity_type,
                    start=candidate.start,
                    end=candidate.end,
                    method=candidate.method,
                    field=field,
                )
            )

        anonymized = self._apply_replacements(safe_text, replacements)
        return anonymized, [entity.to_dict(include_original=include_original) for entity in entities]

    def anonymize_email(
        self,
        email: Dict,
        keep_mapping: bool = False,
        include_original_text: bool = False,
    ) -> Dict:
        """
        Anonymize subject/body and identity-bearing metadata fields.

        Args:
            email: Dict with at least subject/body-like fields.
            keep_mapping: If True, include originals and mapping for controlled
                pseudonymization review. If False, output is irreversible.
            include_original_text: If True, store original subject/body under
                anonymization. Keep False for final dataset exports.
        """
        registry = PlaceholderRegistry()
        subject = "" if email.get("subject") is None else str(email.get("subject", ""))
        body = "" if email.get("body") is None else str(email.get("body", ""))

        anonymized_subject, subject_entities = self.anonymize_text(
            subject,
            field="subject",
            registry=registry,
            include_original=keep_mapping,
        )
        anonymized_body, body_entities = self.anonymize_text(
            body,
            field="body",
            registry=registry,
            include_original=keep_mapping,
        )

        result = dict(email)
        result["subject"] = anonymized_subject
        result["body"] = anonymized_body

        metadata_entities: List[Dict] = []
        original_metadata: Dict[str, object] = {}
        for field in self.SENSITIVE_METADATA_FIELDS:
            if field not in result or result[field] is None:
                continue
            if include_original_text:
                original_metadata[field] = result[field]
            result[field], field_entities = self._anonymize_metadata_value(
                result[field],
                field=field,
                registry=registry,
                include_original=keep_mapping,
            )
            metadata_entities.extend(field_entities)

        anonymization = {
            "entities": subject_entities + body_entities + metadata_entities,
            "mode": "pseudonymize" if keep_mapping else "anonymize",
        }
        if keep_mapping:
            anonymization["mapping"] = registry.export_mapping()
        if include_original_text:
            anonymization["original_text"] = {
                "subject": subject,
                "body": body,
                **original_metadata,
            }

        result["anonymization"] = anonymization
        return result

    def _anonymize_metadata_value(
        self,
        value: object,
        field: str,
        registry: PlaceholderRegistry,
        include_original: bool,
    ) -> Tuple[object, List[Dict]]:
        if isinstance(value, (list, tuple)):
            anonymized_values = []
            entities: List[Dict] = []
            for index, item in enumerate(value):
                anonymized, item_entities = self.anonymize_text(
                    str(item),
                    field=f"{field}[{index}]",
                    registry=registry,
                    include_original=include_original,
                )
                anonymized_values.append(anonymized)
                entities.extend(item_entities)
            return anonymized_values, entities

        return self.anonymize_text(
            str(value),
            field=field,
            registry=registry,
            include_original=include_original,
        )

    def _find_candidates(self, text: str) -> List[EntityCandidate]:
        candidates: List[EntityCandidate] = []
        candidates.extend(self.regex_anonymizer.find(text))
        candidates.extend(self._find_academic_entities(text))
        candidates.extend(self.ner_anonymizer.find(text))
        return [candidate for candidate in candidates if not self._is_whitelisted(candidate.original)]

    def _find_academic_entities(self, text: str) -> List[EntityCandidate]:
        candidates: List[EntityCandidate] = []
        for pattern, entity_type in self.academic_patterns:
            for match in pattern.finditer(text):
                candidates.append(
                    EntityCandidate(
                        match.group(),
                        entity_type,
                        match.start(),
                        match.end(),
                        "ACADEMIC_LIST",
                        85,
                    )
                )
        return candidates

    @staticmethod
    def _resolve_overlaps(candidates: Iterable[EntityCandidate]) -> List[EntityCandidate]:
        """Keep highest-priority, longest non-overlapping candidates."""
        ordered = sorted(
            candidates,
            key=lambda item: (item.priority, item.length, -item.start),
            reverse=True,
        )
        selected: List[EntityCandidate] = []

        for candidate in ordered:
            if candidate.start >= candidate.end:
                continue
            has_overlap = any(
                not (candidate.end <= existing.start or candidate.start >= existing.end)
                for existing in selected
            )
            if not has_overlap:
                selected.append(candidate)

        return sorted(selected, key=lambda item: item.start)

    @staticmethod
    def _apply_replacements(
        text: str,
        replacements: List[Tuple[EntityCandidate, str]],
    ) -> str:
        """Apply replacements from right to left to preserve offsets."""
        anonymized = text
        for candidate, replacement in sorted(
            replacements,
            key=lambda item: item[0].start,
            reverse=True,
        ):
            anonymized = anonymized[: candidate.start] + replacement + anonymized[candidate.end :]
        return anonymized

    @staticmethod
    def _compile_academic_patterns() -> List[Tuple[re.Pattern, str]]:
        patterns: List[Tuple[re.Pattern, str]] = []
        for value in UNIVERSITIES:
            patterns.append((EmailAnonymizer._phrase_pattern(value), "UNIVERSITY"))
        for value in INSTITUTIONS:
            patterns.append((EmailAnonymizer._phrase_pattern(value), "INSTITUTION"))
        return patterns

    @staticmethod
    def _phrase_pattern(phrase: str) -> re.Pattern:
        escaped = re.escape(phrase)
        escaped = escaped.replace(r"\ ", r"\s+")
        return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)

    @staticmethod
    def _is_whitelisted(text: str) -> bool:
        return text.strip().lower() in PLATFORM_WHITELIST


def anonymize_email(email: Dict, keep_mapping: bool = False) -> Dict:
    """Convenience function expected by the pipeline."""
    anonymizer = EmailAnonymizer()
    return anonymizer.anonymize_email(email, keep_mapping=keep_mapping)


def anonymize_text(text: str, keep_mapping: bool = False) -> Dict:
    """Convenience function for standalone text anonymization."""
    anonymizer = EmailAnonymizer()
    registry = PlaceholderRegistry()
    anonymized_text, entities = anonymizer.anonymize_text(
        text,
        registry=registry,
        include_original=keep_mapping,
    )
    result = {
        "anonymized_text": anonymized_text,
        "entities": entities,
        "mode": "pseudonymize" if keep_mapping else "anonymize",
    }
    if keep_mapping:
        result["original_text"] = text
        result["mapping"] = registry.export_mapping()
    return result


def example_usage() -> None:
    """Run a small example from the command line."""
    email = {
        "subject": "Reunião com Ana",
        "body": (
            "Boas Ana, podemos reunir amanhã às 15h no Teams? "
            "O professor João Silva também vai estar. "
            "O meu email é vasco.geada@gmail.com e o meu número é 912345678. "
            "Isto é para a Universidade de Lisboa."
        ),
        "label": "agendamento_reuniao",
    }
    print(json.dumps(anonymize_email(email, keep_mapping=True), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_usage()
