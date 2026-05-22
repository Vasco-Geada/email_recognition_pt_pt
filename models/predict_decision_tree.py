# -*- coding: utf-8 -*-
"""
Inference script for the Decision Tree email classifier.

Examples:
    python models/predict_decision_tree.py --text "Boas Ana, podemos reunir amanha?"
    python models/predict_decision_tree.py --subject "Reuniao" --body "Confirmo as 15h"
    python models/predict_decision_tree.py --file dataset/realistic_emails_v2.json --limit 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.decision_tree_classifier import DecisionTreeEmailClassifier
from models.utils import preprocess_text, save_predictions_to_json


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "decision_tree_model.joblib"
DEFAULT_VECTORIZER = PROJECT_ROOT / "models" / "decision_tree_vectorizer.joblib"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DecisionTreeEmailPredictor:
    """Small inference wrapper around DecisionTreeEmailClassifier."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        vectorizer_path: Path = DEFAULT_VECTORIZER,
    ) -> None:
        self.classifier = DecisionTreeEmailClassifier()
        self.classifier.load(str(model_path), str(vectorizer_path))

    def predict_text(self, text: str) -> Dict:
        """Predict a pre-combined email text."""
        cleaned = preprocess_text(
            text,
            remove_signatures=True,
            remove_threads_history=True,
            remove_punctuation=False,
            remove_stopwords=False,
            lowercase=True,
        )
        if not cleaned:
            return {
                "prediction": None,
                "confidence": 0.0,
                "probabilities": {},
                "error": "empty text",
            }

        result = self.classifier.predict_with_confidence(cleaned)
        result["text_preview"] = text[:120] + ("..." if len(text) > 120 else "")
        return result

    def predict_email(self, subject: str = "", body: str = "") -> Dict:
        """Predict from subject/body fields."""
        text = f"{subject or ''} {body or ''}".strip()
        return self.predict_text(text)

    def predict_file(self, file_path: Path, limit: int | None = None) -> List[Dict]:
        """Predict all emails in a JSON file with subject/body fields."""
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            data = [data]
        if limit is not None:
            data = data[:limit]

        results = []
        for index, email in enumerate(data, start=1):
            if not isinstance(email, dict):
                results.append({"index": index, "error": "item is not an object"})
                continue

            result = self.predict_email(
                subject=str(email.get("subject", "") or ""),
                body=str(email.get("body", "") or ""),
            )
            result["index"] = index
            if "label" in email:
                result["gold_label"] = email["label"]
            results.append(result)

        return results


def print_result(result: Dict) -> None:
    """Pretty-print a prediction result as JSON."""
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict email intent with the Decision Tree baseline"
    )
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--subject", type=str, default="")
    parser.add_argument("--body", type=str, default="")
    parser.add_argument("--file", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vectorizer", type=Path, default=DEFAULT_VECTORIZER)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = DecisionTreeEmailPredictor(args.model, args.vectorizer)

    if args.file:
        results = predictor.predict_file(args.file, limit=args.limit)
        if args.output:
            save_predictions_to_json(results, str(args.output))
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    if args.text:
        result = predictor.predict_text(args.text)
    else:
        result = predictor.predict_email(args.subject, args.body)

    print_result(result)


if __name__ == "__main__":
    main()

