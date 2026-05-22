# -*- coding: utf-8 -*-
"""
Evaluate the Decision Tree baseline on dataset/realistic_emails_v2.json.

Default usage:
    python models/evaluate_decision_tree.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.decision_tree_classifier import DecisionTreeEmailClassifier
from models.train_decision_tree import (
    DEFAULT_DATASET,
    load_and_validate_dataset,
    prepare_texts_and_labels,
)
from models.utils import preprocess_texts


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_decision_tree(
    dataset_path: Path = DEFAULT_DATASET,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Dict:
    """Train/evaluate Decision Tree and return a report dictionary."""
    emails = load_and_validate_dataset(dataset_path)
    texts, labels = prepare_texts_and_labels(emails)
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

    classifier = DecisionTreeEmailClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    classifier.fit(x_train, y_train)
    metrics = classifier.evaluate(x_test, y_test, verbose=True)

    cv_pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents=None,
                    ngram_range=(1, 2),
                    max_features=5000,
                ),
            ),
            (
                "tree",
                DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state,
                ),
            ),
        ]
    )
    cv_scores = cross_val_score(cv_pipeline, texts, labels, cv=5, scoring="f1_weighted")

    report = {
        "dataset_path": str(dataset_path),
        "total_samples": len(texts),
        "train_size": len(x_train),
        "test_size": len(x_test),
        "params": {
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
        },
        "accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"],
        "cv_f1_weighted_mean": float(cv_scores.mean()),
        "cv_f1_weighted_std": float(cv_scores.std()),
        "classes": metrics["classes"],
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report_dict"],
        "feature_importance": classifier.get_feature_importance(top_n=20),
    }

    logger.info("5-fold weighted F1: %.4f (+/- %.4f)", cv_scores.mean(), cv_scores.std())
    return report


def print_comparison_guidance(report: Dict) -> None:
    """Print dissertation-oriented comparison guidance."""
    print("\n=== Decision Tree baseline summary ===")
    print(f"Dataset: {report['dataset_path']}")
    print(f"Samples: {report['total_samples']}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Macro F1: {report['f1_macro']:.4f}")
    print(f"Weighted F1: {report['f1_weighted']:.4f}")
    print(
        "CV weighted F1: "
        f"{report['cv_f1_weighted_mean']:.4f} "
        f"(+/- {report['cv_f1_weighted_std']:.4f})"
    )
    print("\nUse these three values for model comparison tables:")
    print("- accuracy: global correctness")
    print("- macro F1: treats each class equally")
    print("- weighted F1: accounts for class distribution")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Decision Tree baseline for PT-PT email intents"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_decision_tree(
        dataset_path=args.dataset,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        test_size=args.test_size,
    )
    print_comparison_guidance(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
        logger.info("Saved evaluation report to %s", args.output)


if __name__ == "__main__":
    main()
