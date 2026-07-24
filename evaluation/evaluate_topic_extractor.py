"""Evaluate the classic topic extractor on an independent labelled dataset.

Run from the project root:

    python evaluation/evaluate_topic_extractor.py \
        --dataset dataset/topic_extraction_dev.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.argument_extraction import TopicExtractor


def normalize_text(value: str) -> str:
    """Normalize a topic for case-, accent- and punctuation-insensitive EM."""
    decomposed = unicodedata.normalize("NFKD", str(value or "").lower())
    without_accents = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )
    tokens = re.findall(r"[a-z0-9]+", without_accents)
    return " ".join(tokens)


def token_scores(gold: str, predicted: str) -> Tuple[float, float, float]:
    """Return bag-of-words precision, recall and F1 for one positive example."""
    gold_tokens = normalize_text(gold).split()
    predicted_tokens = normalize_text(predicted).split()
    if not gold_tokens:
        return (1.0, 1.0, 1.0) if not predicted_tokens else (0.0, 0.0, 0.0)
    if not predicted_tokens:
        return 0.0, 0.0, 0.0

    overlap = sum((Counter(gold_tokens) & Counter(predicted_tokens)).values())
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def load_cases(path: Path, split: str | None = None) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise ValueError("O dataset deve ser uma lista ou um objeto com a chave 'cases'.")

    required = {"id", "subject", "body", "label", "gold_topic"}
    validated = []
    seen_ids = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Caso {index} nao e um objeto JSON.")
        missing = required.difference(case)
        if missing:
            raise ValueError(
                f"Caso {case.get('id', index)!r} sem campos: {sorted(missing)}"
            )
        if case["id"] in seen_ids:
            raise ValueError(f"ID duplicado: {case['id']!r}")
        seen_ids.add(case["id"])
        if split and case.get("split") != split:
            continue
        validated.append(case)

    if not validated:
        raise ValueError("O filtro selecionado nao contem casos para avaliar.")
    return validated


def evaluate_cases(
    cases: Sequence[Dict[str, Any]],
    extractor: TopicExtractor | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    extractor = extractor or TopicExtractor(nlp_model=None)
    rows: List[Dict[str, Any]] = []

    for case in cases:
        spans = extractor.extract(
            text=case["body"],
            subject=case["subject"],
            intent=case["label"],
        )
        predicted = spans[0].text if spans else ""
        gold = str(case["gold_topic"] or "")
        gold_normalized = normalize_text(gold)
        predicted_normalized = normalize_text(predicted)
        is_positive = bool(gold_normalized)
        exact_match = gold_normalized == predicted_normalized
        precision, recall, f1 = token_scores(gold, predicted)

        if exact_match:
            error_type = "exact_match" if is_positive else "true_negative"
        elif not is_positive:
            error_type = "false_positive"
        elif not predicted_normalized:
            error_type = "false_negative"
        else:
            error_type = "boundary_or_topic_mismatch"

        rows.append(
            {
                "id": case["id"],
                "split": case.get("split", ""),
                "label": case["label"],
                "subject": case["subject"],
                "body": case["body"],
                "gold_topic": gold,
                "predicted_topic": predicted,
                "exact_match": exact_match,
                "token_precision": precision,
                "token_recall": recall,
                "token_f1": f1,
                "error_type": error_type,
                "confidence": spans[0].confidence if spans else None,
                "extraction_method": spans[0].extraction_method if spans else "",
            }
        )

    positives = [row for row in rows if normalize_text(row["gold_topic"])]
    negatives = [row for row in rows if not normalize_text(row["gold_topic"])]
    extracted = [row for row in rows if normalize_text(row["predicted_topic"])]
    detection_tp = sum(
        bool(normalize_text(row["gold_topic"]))
        and bool(normalize_text(row["predicted_topic"]))
        for row in rows
    )
    detection_fp = sum(row["error_type"] == "false_positive" for row in rows)
    detection_fn = sum(row["error_type"] == "false_negative" for row in rows)
    detection_precision = (
        detection_tp / (detection_tp + detection_fp)
        if detection_tp + detection_fp
        else 0.0
    )
    detection_recall = (
        detection_tp / (detection_tp + detection_fn)
        if detection_tp + detection_fn
        else 0.0
    )
    detection_f1 = (
        2 * detection_precision * detection_recall
        / (detection_precision + detection_recall)
        if detection_precision + detection_recall
        else 0.0
    )

    metrics = {
        "total": len(rows),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "overall_exact_match": _mean(row["exact_match"] for row in rows),
        "positive_exact_match": _mean(row["exact_match"] for row in positives),
        "positive_token_precision": _mean(
            row["token_precision"] for row in positives
        ),
        "positive_token_recall": _mean(row["token_recall"] for row in positives),
        "positive_token_f1": _mean(row["token_f1"] for row in positives),
        "negative_accuracy": _mean(row["exact_match"] for row in negatives),
        "false_positive_rate": _mean(
            row["error_type"] == "false_positive" for row in negatives
        ),
        "extraction_coverage": len(extracted) / len(rows),
        "topic_detection_precision": detection_precision,
        "topic_detection_recall": detection_recall,
        "topic_detection_f1": detection_f1,
        "error_counts": dict(Counter(row["error_type"] for row in rows)),
    }
    return metrics, rows


def write_results(
    output_dir: Path,
    dataset_path: Path,
    metrics: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "topic_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"dataset": str(dataset_path), "metrics": metrics},
            handle,
            ensure_ascii=False,
            indent=2,
        )

    fieldnames = list(rows[0].keys())
    with (output_dir / "topic_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(metrics: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> None:
    print("\n=== Avaliacao independente do TopicExtractor ===")
    print(
        f"Casos: {metrics['total']} "
        f"({metrics['positive_count']} com topico, "
        f"{metrics['negative_count']} sem topico)"
    )
    print(f"Overall exact match:       {metrics['overall_exact_match']:.3f}")
    print(f"Positive exact match:      {metrics['positive_exact_match']:.3f}")
    print(f"Positive token F1:         {metrics['positive_token_f1']:.3f}")
    print(f"Negative accuracy:         {metrics['negative_accuracy']:.3f}")
    print(f"False-positive rate:       {metrics['false_positive_rate']:.3f}")
    print(f"Topic detection F1:        {metrics['topic_detection_f1']:.3f}")

    errors = [row for row in rows if row["error_type"] not in {
        "exact_match",
        "true_negative",
    }]
    if errors:
        print("\nPrimeiros erros:")
        for row in errors[:8]:
            print(
                f"- {row['id']} [{row['error_type']}]: "
                f"gold={row['gold_topic']!r} | pred={row['predicted_topic']!r}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia o TopicExtractor em exemplos rotulados."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases(args.dataset, split=args.split)
    metrics, rows = evaluate_cases(cases)
    print_summary(metrics, rows)
    if args.output_dir:
        write_results(args.output_dir, args.dataset, metrics, rows)
        print(f"\nResultados guardados em: {args.output_dir}")


if __name__ == "__main__":
    main()
