import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.temporal_normalization import TemporalNormalizer


MEETING_LABELS = {
    "agendamento_reuniao",
    "cancelamento_reuniao",
    "reuniao_confirmada",
}

REMOTE_LOCATION_PATTERN = re.compile(
    r"\b(zoom|teams|microsoft\s+teams|google\s+meet|meet|online|remot[ao])\b",
    re.IGNORECASE,
)
ROOM_DEPARTMENT_LOCATION_PATTERN = re.compile(
    r"\bsala\s+de\s+reuni(?:ao|oes|ões|Ãµes)\s+do\s+departamento\b",
    re.IGNORECASE,
)
ROOM_LOCATION_PATTERN = re.compile(
    r"\bsala(?:\s+de\s+reuni(?:ao|oes|ões))?(?:\s+[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)?)?(?:\s+do\s+departamento)?\b",
    re.IGNORECASE,
)
OFFICE_LOCATION_PATTERN = re.compile(
    r"\bgabinete(?:\s+(?:da\s+direcao|da\s+direção|de\s+\w+|\d+(?:[.\-]\d+)?))?\b",
    re.IGNORECASE,
)
PRESENTIAL_LOCATION_PATTERN = re.compile(
    r"\b(secretaria(?!\s+academica)|biblioteca|laborat(?:orio|ório)(?:\s+de\s+\w+)?|audit(?:orio|ório)(?:\s+\w+)?|campus(?:\s+\w+)?)\b",
    re.IGNORECASE,
)

PARTICIPANT_STOPWORDS = {
    "boa",
    "bom",
    "ola",
    "olá",
    "viva",
    "exmo",
    "exma",
    "caro",
    "cara",
    "professor",
    "professora",
    "obrigado",
    "obrigada",
    "cumprimentos",
    "enviado",
    "enviado do outlook",
    "enviado do telemovel",
    "sent from my iphone",
    "mensagem enviada automaticamente",
    "proponho",
    "preciso",
}

TITLE_PREFIXES = (
    "prof.",
    "prof",
    "professor",
    "professora",
    "dr.",
    "dra.",
    "dr",
    "dra",
)


def as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def canonical_participant(value: Any) -> str:
    text = " ".join(str(value or "").strip().lower().replace(".", " ").split())
    tokens = text.split()
    while tokens and tokens[0] in {prefix.replace(".", "") for prefix in TITLE_PREFIXES}:
        tokens = tokens[1:]
    return " ".join(tokens)


def unique_clean(values: List[Any]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = canonical_participant(text)
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def load_spacy_model() -> Optional[Any]:
    try:
        import spacy

        return spacy.load("pt_core_news_sm")
    except Exception:
        return None


def ner_participants(text: str, nlp: Optional[Any]) -> List[str]:
    if nlp is None or not text:
        return []

    doc = nlp(text)
    names = []
    for ent in doc.ents:
        if ent.label_ in {"PER", "PERSON"}:
            candidate = clean_ner_participant(ent.text)
            if is_valid_ner_participant(candidate):
                names.append(candidate)
    return unique_clean(names)


def clean_ner_participant(candidate: str) -> str:
    text = " ".join(str(candidate or "").strip().split())
    text = re_sub_prefix(text, [
        r"ola\s+",
        r"olá\s+",
        r"car[ao]\s+colega\s+",
        r"car[ao]\s+",
        r"estimad[oa]/a\s+estudante\s+",
        r"estimad[oa]\s+",
        r"exmo\.?\s+",
        r"exma\.?\s+",
    ])
    return text.strip(" ,.;:")


def re_sub_prefix(text: str, patterns: List[str]) -> str:
    import re

    for pattern in patterns:
        text = re.sub(rf"^(?:{pattern})", "", text, flags=re.IGNORECASE)
    return text


def is_valid_ner_participant(candidate: str) -> bool:
    normalized = " ".join(candidate.lower().split())
    canonical = canonical_participant(candidate)
    if not canonical:
        return False
    if normalized in PARTICIPANT_STOPWORDS or canonical in PARTICIPANT_STOPWORDS:
        return False
    if any(stopword in normalized for stopword in ["enviado", "outlook", "iphone", "telemovel", "automatica"]):
        return False
    # Single-token NER is too noisy in this email domain unless it is metadata,
    # and metadata participants are already included separately.
    if len(canonical.split()) < 2:
        return False
    return True


def metadata_participants(email: Dict[str, Any]) -> List[str]:
    values: List[Any] = []
    values.extend(as_list(email.get("participants")))
    values.extend(as_list(email.get("sender")))
    values.extend(as_list(email.get("recipient")))
    return unique_clean(values)


def build_participants(email: Dict[str, Any], nlp: Optional[Any]) -> List[str]:
    text = " ".join(
        str(email.get(key, "") or "")
        for key in ("subject", "body")
    )
    return unique_clean(metadata_participants(email) + ner_participants(text, nlp))


def parse_reference_datetime(email: Dict[str, Any]) -> datetime:
    for key in ("sent_datetime", "sent_at", "email_date", "date", "created_at"):
        value = email.get(key)
        if not value:
            continue
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            continue
    return datetime.now()


def expression_in_email(email: Dict[str, Any], expression: str) -> bool:
    haystack = " ".join(
        str(email.get(key, "") or "")
        for key in ("subject", "body")
    ).lower()
    needle = str(expression or "").strip().lower()
    return bool(needle and needle in haystack)


def build_time_arguments(
    email: Dict[str, Any],
    normalizer: TemporalNormalizer,
) -> Dict[str, List[str]]:
    expression = str(email.get("time_expression") or "").strip()
    if not expression or not expression_in_email(email, expression):
        return {"time": [], "time_normalized": []}

    reference_datetime = parse_reference_datetime(email)
    normalized = normalizer.normalize(
        expression,
        reference_datetime=reference_datetime,
    ).to_dict()

    normalized_values = []
    for key in ("normalized_datetime", "interval_start"):
        value = normalized.get(key)
        if value:
            normalized_values.append(str(value))

    return {
        "time": [expression],
        "time_normalized": normalized_values,
    }


def normalize_location(location: str) -> Optional[str]:
    text = str(location or "").strip()
    if not text:
        return None

    text = re.sub(r"^\s*(?:em|no|na|por|via)\s+", "", text, flags=re.IGNORECASE)
    remote_match = REMOTE_LOCATION_PATTERN.search(text)
    if remote_match:
        return f"remoto - {normalize_remote_platform(remote_match.group(0))}"
    room_department_match = ROOM_DEPARTMENT_LOCATION_PATTERN.search(text)
    if room_department_match:
        return f"presencial - {room_department_match.group(0).strip()}"
    room_match = ROOM_LOCATION_PATTERN.search(text)
    if room_match:
        return f"presencial - {room_match.group(0).strip()}"
    office_match = OFFICE_LOCATION_PATTERN.search(text)
    if office_match:
        return f"presencial - {office_match.group(0).strip()}"
    presential_match = PRESENTIAL_LOCATION_PATTERN.search(text)
    if presential_match:
        return f"presencial - {presential_match.group(0).strip()}"
    return None


def normalize_remote_platform(platform: str) -> str:
    text = str(platform or "").strip().lower()
    if "team" in text:
        return "teams"
    if "meet" in text:
        return "google meet"
    if "zoom" in text:
        return "zoom"
    return "zoom"


def explicit_location_from_text(email: Dict[str, Any]) -> Optional[str]:
    text = str(email.get("body", "") or "")
    for pattern in (
        REMOTE_LOCATION_PATTERN,
        ROOM_DEPARTMENT_LOCATION_PATTERN,
        ROOM_LOCATION_PATTERN,
        OFFICE_LOCATION_PATTERN,
        PRESENTIAL_LOCATION_PATTERN,
    ):
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start() - 10):match.start()]
            if re.search(r"(?:em|no|na|por|via)\s+$", prefix, re.IGNORECASE):
                return match.group(0).strip()
    return None


def inferred_location(email: Dict[str, Any]) -> List[str]:
    label = str(email.get("label", ""))
    if label not in MEETING_LABELS:
        return []

    explicit_location = explicit_location_from_text(email)
    normalized = normalize_location(explicit_location or "")
    return [normalized or "remoto - zoom"]


def build_gold_item(
    email: Dict[str, Any],
    index: int,
    nlp: Optional[Any],
    normalizer: TemporalNormalizer,
) -> Dict[str, Any]:
    label = str(email.get("label", ""))
    is_meeting = label in MEETING_LABELS
    time_arguments = build_time_arguments(email, normalizer) if is_meeting else {
        "time": [],
        "time_normalized": [],
    }

    arguments = {
        "participants": build_participants(email, nlp),
        "time": time_arguments["time"],
        "time_normalized": time_arguments["time_normalized"],
        "location": inferred_location(email),
        "topic": as_list(email.get("topic")) if is_meeting else [],
    }

    return {
        "id": index,
        "dataset_id": email.get("id"),
        "subject": email.get("subject", ""),
        "text": email.get("body", ""),
        "sent_datetime": email.get("sent_datetime"),
        "intent": label,
        "trigger": [],
        "arguments": arguments,
        "confidence": {
            "trigger": 1.0,
            "participants": 1.0,
            "temporal": 1.0,
            "location": 1.0,
            "topic": 1.0,
        },
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "validated": True,
            "version": "metadata_gold_v2",
            "source": "dataset metadata + sent_datetime temporal normalization",
            "sent_datetime": email.get("sent_datetime"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate aligned gold annotations from dataset metadata."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    with dataset_path.open("r", encoding="utf-8") as file:
        emails = json.load(file)

    nlp = load_spacy_model()
    if nlp is None:
        print("Aviso: spaCy pt_core_news_sm indisponivel; gold usa apenas sender/recipient/participants.")

    normalizer = TemporalNormalizer()
    gold = [
        build_gold_item(email, index, nlp, normalizer)
        for index, email in enumerate(emails)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(gold, file, ensure_ascii=False, indent=2)

    print(f"{len(gold)} gold annotations geradas.")
    print(f"Guardado em: {output_path}")


if __name__ == "__main__":
    main()
