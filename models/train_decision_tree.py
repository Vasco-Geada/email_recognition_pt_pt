# -*- coding: utf-8 -*-
"""
Train the Decision Tree email intent classifier.

Default usage from the project root:
    python models/train_decision_tree.py

The default dataset is always:
    dataset/realistic_emails_v2.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.decision_tree_classifier import DecisionTreeEmailClassifier
from models.utils import preprocess_texts


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "realistic_emails_v2.json"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "decision_tree_model.joblib"
DEFAULT_VECTORIZER = PROJECT_ROOT / "models" / "decision_tree_vectorizer.joblib"
REQUIRED_FIELDS = ("subject", "body", "label")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_and_validate_dataset(dataset_path: Path) -> List[Dict]:
    """Load JSON dataset and validate required fields."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list of email objects")

    valid_emails: List[Dict] = []
    skipped = 0
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            logger.warning("Skipping item %s: expected object", index)
            skipped += 1
            continue

        missing = [field for field in REQUIRED_FIELDS if field not in item]
        if missing:
            logger.warning("Skipping item %s: missing fields %s", index, missing)
            skipped += 1
            continue

        subject = "" if item.get("subject") is None else str(item.get("subject")).strip()
        body = "" if item.get("body") is None else str(item.get("body")).strip()
        label = "" if item.get("label") is None else str(item.get("label")).strip()

        if not label:
            logger.warning("Skipping item %s: empty label", index)
            skipped += 1
            continue
        if not subject and not body:
            logger.warning("Skipping item %s: empty subject and body", index)
            skipped += 1
            continue

        valid_emails.append({"subject": subject, "body": body, "label": label})

    if not valid_emails:
        raise ValueError("No valid emails found in dataset")

    logger.info("Loaded %s valid emails from %s", len(valid_emails), dataset_path)
    if skipped:
        logger.warning("Skipped %s invalid/empty items", skipped)
    return valid_emails


def prepare_texts_and_labels(emails: List[Dict]) -> Tuple[List[str], List[str]]:
    """Combine subject + body into text and collect labels."""
    texts: List[str] = []
    labels: List[str] = []

    for email in emails:
        text = f"{email['subject']} {email['body']}".strip()
        if text:
            texts.append(text)
            labels.append(email["label"])

    return texts, labels


def log_class_distribution(labels: List[str]) -> None:
    """Log dataset class distribution."""
    total = len(labels)
    for label in sorted(set(labels)):
        count = labels.count(label)
        logger.info("  - %s: %s (%.1f%%)", label, count, (count / total) * 100)


def train_classifier(
    dataset_path: Path = DEFAULT_DATASET,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[DecisionTreeEmailClassifier, Dict]:
    """Train and evaluate the Decision Tree classifier."""
    logger.info("=" * 80)
    logger.info("TRAINING DECISION TREE EMAIL CLASSIFIER")
    logger.info("=" * 80)
    logger.info("Dataset: %s", dataset_path)

    emails = load_and_validate_dataset(dataset_path)
    texts, labels = prepare_texts_and_labels(emails)

    logger.info("Class distribution:")
    log_class_distribution(labels)

    texts = preprocess_texts(
        texts,
        remove_signatures=True,
        remove_threads_history=True,
        remove_punctuation=False,
        remove_stopwords=False,
        lowercase=True,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    logger.info("Train samples: %s", len(x_train))
    logger.info("Test samples: %s", len(x_test))

    classifier = DecisionTreeEmailClassifier(
        max_features=max_features,
        ngram_range=ngram_range,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    classifier.fit(x_train, y_train)

    metrics = classifier.evaluate(x_test, y_test, verbose=verbose)
    metrics["train_size"] = len(x_train)
    metrics["test_size"] = len(x_test)
    metrics["dataset_path"] = str(dataset_path)
    metrics["model_params"] = {
        "max_features": max_features,
        "ngram_range": ngram_range,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "random_state": random_state,
    }
    metrics["feature_importance"] = classifier.get_feature_importance(top_n=20)

    logger.info("Top Decision Tree features:")
    for feature, score in metrics["feature_importance"][:10]:
        logger.info("  - %s: %.4f", feature, score)

    logger.info("=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("Accuracy: %.4f", metrics["accuracy"])
    logger.info("Macro F1: %.4f", metrics["f1_macro"])
    logger.info("Weighted F1: %.4f", metrics["f1_weighted"])
    logger.info("=" * 80)
    return classifier, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Decision Tree baseline for PT-PT email intents"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--ngrams", type=int, nargs=2, default=[1, 2])
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-vectorizer", type=Path, default=DEFAULT_VECTORIZER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classifier, _ = train_classifier(
        dataset_path=args.dataset,
        max_features=args.max_features,
        ngram_range=tuple(args.ngrams),
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        test_size=args.test_size,
        random_state=args.random_state,
        verbose=True,
    )
    classifier.save(str(args.output_model), str(args.output_vectorizer))
    logger.info("Decision Tree training completed successfully")


if __name__ == "__main__":
    main()

