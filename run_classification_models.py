# -*- coding: utf-8 -*-
"""Train all intent classifiers once and evaluate them on a separate dataset.

Examples:
    python run_classification_models.py train \
        --dataset dataset/train.json

    python run_classification_models.py evaluate \
        --dataset dataset/test.json \
        --model-dir trained_models/email_intent \
        --output-dir evaluation_results/independent_classification
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from models.decision_tree_classifier import DecisionTreeEmailClassifier
from models.logistic_regression_classifier import (
    LogisticRegressionEmailClassifier,
)
from models.naive_bayes_classifier import NaiveBayesEmailClassifier
from run_project_evaluation import (
    PreparedDataset,
    build_classification_predictions,
    build_error_analysis,
    prepare_dataset,
    write_confusion_matrix_csv,
    write_error_analysis_csv,
    write_json,
    write_summary_csv,
)


logger = logging.getLogger("run_classification_models")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "trained_models" / "email_intent"

EXPECTED_LABELS = {
    "agendamento_reuniao",
    "cancelamento_reuniao",
    "reuniao_confirmada",
    "nao_reuniao",
}
MODEL_NAMES = (
    "logistic_regression",
    "naive_bayes",
    "decision_tree",
)
METADATA_FILENAME = "training_metadata.json"


def model_artifact_paths(
    model_dir: Path,
    model_name: str,
) -> Tuple[Path, Path]:
    return (
        model_dir / f"{model_name}_model.joblib",
        model_dir / f"{model_name}_vectorizer.joblib",
    )


def create_models(
    max_features: int = 5000,
    random_state: int = 42,
) -> Dict[str, Any]:
    return {
        "logistic_regression": LogisticRegressionEmailClassifier(
            max_features=max_features,
            random_state=random_state,
        ),
        "naive_bayes": NaiveBayesEmailClassifier(
            max_features=max_features,
            random_state=random_state,
        ),
        "decision_tree": DecisionTreeEmailClassifier(
            max_features=max_features,
            random_state=random_state,
        ),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_fingerprints(texts: List[str]) -> List[str]:
    return sorted({
        hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        for text in texts
        if text.strip()
    })


def validate_training_labels(labels: List[str]) -> None:
    observed = set(labels)
    missing = EXPECTED_LABELS.difference(observed)
    if missing:
        raise ValueError(
            "O dataset de treino nao contem todas as quatro classes. "
            f"Em falta: {sorted(missing)}"
        )


def _save_classifier(
    classifier: Any,
    model_path: Path,
    vectorizer_path: Path,
) -> None:
    classifier.save(str(model_path), str(vectorizer_path))


def _load_classifier(
    model_name: str,
    model_path: Path,
    vectorizer_path: Path,
) -> Any:
    if model_name == "logistic_regression":
        return LogisticRegressionEmailClassifier().load(
            str(model_path),
            str(vectorizer_path),
        )
    if model_name == "naive_bayes":
        classifier = NaiveBayesEmailClassifier()
        classifier.load(str(model_path), str(vectorizer_path))
        return classifier
    if model_name == "decision_tree":
        return DecisionTreeEmailClassifier().load(
            str(model_path),
            str(vectorizer_path),
        )
    raise ValueError(f"Modelo desconhecido: {model_name}")


def _feature_importance(classifier: Any) -> Any:
    if isinstance(classifier, DecisionTreeEmailClassifier):
        return classifier.get_feature_importance(top_n=20)
    return classifier.get_feature_importance(top_n=20)


def train_all_models(
    dataset_path: Path,
    model_dir: Path,
    use_anonymization: bool = False,
    max_features: int = 5000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Fit all classifiers on the complete training dataset and persist them."""
    dataset_path = dataset_path.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_dataset(
        str(dataset_path),
        use_anonymization=use_anonymization,
    )
    validate_training_labels(prepared.labels)

    models = create_models(
        max_features=max_features,
        random_state=random_state,
    )
    model_summaries: Dict[str, Any] = {}
    for model_name, classifier in models.items():
        logger.info("A treinar %s com %s emails", model_name, len(prepared.texts))
        started_at = time.perf_counter()
        classifier.fit(prepared.texts, prepared.labels)
        train_seconds = time.perf_counter() - started_at
        model_path, vectorizer_path = model_artifact_paths(model_dir, model_name)
        _save_classifier(classifier, model_path, vectorizer_path)
        model_summaries[model_name] = {
            "classes": [str(label) for label in classifier.model.classes_],
            "vocabulary_size": len(classifier.vectorizer.vocabulary_),
            "train_seconds": train_seconds,
            "model_file": model_path.name,
            "vectorizer_file": vectorizer_path.name,
            "top_features": _feature_importance(classifier),
        }

    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_dataset": str(dataset_path),
        "training_dataset_sha256": file_sha256(dataset_path),
        "training_text_fingerprints": text_fingerprints(prepared.texts),
        "num_training_examples": len(prepared.texts),
        "class_distribution": dict(sorted(Counter(prepared.labels).items())),
        "labels": sorted(set(prepared.labels)),
        "preprocessing": {
            "use_anonymization": use_anonymization,
            "subject_weight": 1.0,
            "body_weight": 1.0,
            "remove_punctuation": False,
            "lowercase": True,
        },
        "model_config": {
            "max_features": max_features,
            "ngram_range": [1, 2],
            "random_state": random_state,
        },
        "models": model_summaries,
    }
    write_json(model_dir / METADATA_FILENAME, metadata)
    return metadata


def load_training_metadata(model_dir: Path) -> Dict[str, Any]:
    metadata_path = model_dir / METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadados de treino nao encontrados: {metadata_path}"
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("schema_version") != 1:
        raise ValueError(
            f"Versao de metadados nao suportada: "
            f"{metadata.get('schema_version')!r}"
        )
    return metadata


def load_all_models(model_dir: Path) -> Dict[str, Any]:
    models = {}
    for model_name in MODEL_NAMES:
        model_path, vectorizer_path = model_artifact_paths(model_dir, model_name)
        models[model_name] = _load_classifier(
            model_name,
            model_path,
            vectorizer_path,
        )
    return models


def validate_independent_evaluation(
    dataset_path: Path,
    prepared: PreparedDataset,
    metadata: Dict[str, Any],
    allow_overlap: bool,
) -> Dict[str, Any]:
    dataset_hash = file_sha256(dataset_path)
    training_hash = metadata.get("training_dataset_sha256")
    training_fingerprints = set(
        metadata.get("training_text_fingerprints", [])
    )
    evaluation_fingerprints = set(text_fingerprints(prepared.texts))
    overlap_count = len(training_fingerprints & evaluation_fingerprints)

    if not allow_overlap and dataset_hash == training_hash:
        raise ValueError(
            "O dataset de avaliacao e exatamente o mesmo usado no treino. "
            "Escolhe outro dataset ou usa --allow-overlap conscientemente."
        )
    if not allow_overlap and overlap_count:
        raise ValueError(
            f"Foram encontrados {overlap_count} emails do treino no dataset "
            "de avaliacao. Remove a sobreposicao ou usa --allow-overlap."
        )

    trained_labels = set(metadata.get("labels", []))
    unknown_labels = set(prepared.labels).difference(trained_labels)
    if unknown_labels:
        raise ValueError(
            "O dataset de avaliacao contem classes ausentes do treino: "
            f"{sorted(unknown_labels)}"
        )

    return {
        "evaluation_dataset_sha256": dataset_hash,
        "overlapping_email_count": overlap_count,
        "overlap_allowed": allow_overlap,
    }


def evaluate_loaded_classifier(
    model_name: str,
    classifier: Any,
    prepared: PreparedDataset,
) -> Tuple[Dict[str, Any], List[str], List[Dict[str, float]]]:
    started_at = time.perf_counter()
    raw_predictions = classifier.predict(prepared.texts)
    predict_seconds = time.perf_counter() - started_at
    raw_probabilities = classifier.predict_proba(prepared.texts)

    predictions = (
        [str(raw_predictions)]
        if isinstance(raw_predictions, str)
        else [str(value) for value in raw_predictions]
    )
    probabilities = (
        [raw_probabilities]
        if isinstance(raw_probabilities, dict)
        else raw_probabilities
    )
    labels = sorted(
        set(prepared.labels)
        | set(predictions)
        | {str(label) for label in classifier.model.classes_}
    )
    metrics = {
        "model": model_name,
        "num_test": len(prepared.texts),
        "predict_seconds": predict_seconds,
        "accuracy": accuracy_score(prepared.labels, predictions),
        "precision_macro": precision_score(
            prepared.labels,
            predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            prepared.labels,
            predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            prepared.labels,
            predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "f1_weighted": f1_score(
            prepared.labels,
            predictions,
            labels=labels,
            average="weighted",
            zero_division=0,
        ),
        "labels": labels,
        "confusion_matrix": confusion_matrix(
            prepared.labels,
            predictions,
            labels=labels,
        ).tolist(),
        "classification_report": classification_report(
            prepared.labels,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics, predictions, probabilities


def evaluate_saved_models(
    dataset_path: Path,
    model_dir: Path,
    output_dir: Path,
    allow_overlap: bool = False,
) -> Dict[str, Any]:
    """Load persisted classifiers and evaluate without fitting any component."""
    dataset_path = dataset_path.resolve()
    metadata = load_training_metadata(model_dir)
    use_anonymization = bool(
        metadata.get("preprocessing", {}).get("use_anonymization", True)
    )
    prepared = prepare_dataset(
        str(dataset_path),
        use_anonymization=use_anonymization,
    )
    independence = validate_independent_evaluation(
        dataset_path,
        prepared,
        metadata,
        allow_overlap=allow_overlap,
    )
    models = load_all_models(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: Dict[str, Dict[str, Any]] = {}
    for model_name, classifier in models.items():
        logger.info("A avaliar %s sem novo treino", model_name)
        metrics, predictions, probabilities = evaluate_loaded_classifier(
            model_name,
            classifier,
            prepared,
        )
        rows = build_classification_predictions(
            test_emails=prepared.processed_emails,
            test_texts=prepared.texts,
            y_test=prepared.labels,
            predictions=predictions,
            probabilities=probabilities,
        )
        all_metrics[model_name] = metrics
        write_json(output_dir / f"{model_name}_metrics.json", metrics)
        write_json(output_dir / f"{model_name}_predictions.json", rows)
        write_confusion_matrix_csv(
            output_dir / f"{model_name}_confusion_matrix.csv",
            metrics["labels"],
            metrics["confusion_matrix"],
        )
        error_analysis = build_error_analysis(rows)
        write_json(
            output_dir / f"{model_name}_error_analysis.json",
            error_analysis,
        )
        write_error_analysis_csv(
            output_dir / f"{model_name}_errors.csv",
            error_analysis["errors"],
        )

    write_summary_csv(output_dir / "summary.csv", all_metrics)
    evaluation_summary = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "training_dataset": metadata.get("training_dataset"),
        "evaluation_dataset": str(dataset_path),
        "model_dir": str(model_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "num_test": len(prepared.texts),
        "class_distribution": dict(sorted(Counter(prepared.labels).items())),
        **independence,
        "metrics": all_metrics,
    }
    write_json(output_dir / "summary.json", evaluation_summary)
    return evaluation_summary


def print_training_summary(metadata: Dict[str, Any], model_dir: Path) -> None:
    print("\n=== Treino concluido ===")
    print(f"Emails de treino: {metadata['num_training_examples']}")
    print(f"Classes: {', '.join(metadata['labels'])}")
    for model_name, details in metadata["models"].items():
        print(
            f"- {model_name}: {details['vocabulary_size']} features, "
            f"{details['train_seconds']:.3f}s"
        )
    print(f"Modelos guardados em: {model_dir.resolve()}")


def print_evaluation_summary(summary: Dict[str, Any]) -> None:
    print("\n=== Avaliacao independente concluida ===")
    print(f"Emails de teste: {summary['num_test']}")
    print(f"Sobreposicao com treino: {summary['overlapping_email_count']}")
    for model_name, metrics in summary["metrics"].items():
        print(
            f"- {model_name}: accuracy={metrics['accuracy']:.4f}, "
            f"f1_macro={metrics['f1_macro']:.4f}"
        )
    print(f"Resultados guardados em: {summary['output_dir']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Treina os classificadores de intencao e avalia-os num dataset "
            "independente."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Treina os tres modelos com todo o dataset indicado.",
    )
    train_parser.add_argument("--dataset", type=Path, required=True)
    train_parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
    )
    train_parser.add_argument("--max-features", type=int, default=5000)
    train_parser.add_argument("--random-state", type=int, default=42)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Carrega os modelos e avalia-os sem voltar a treinar.",
    )
    evaluate_parser.add_argument("--dataset", type=Path, required=True)
    evaluate_parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
    )
    evaluate_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results/independent_classification"),
    )
    evaluate_parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Permite conscientemente emails de treino no dataset de teste.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    if args.command == "train":
        metadata = train_all_models(
            dataset_path=args.dataset,
            model_dir=args.model_dir,
            use_anonymization=False,
            max_features=args.max_features,
            random_state=args.random_state,
        )
        print_training_summary(metadata, args.model_dir)
        return

    summary = evaluate_saved_models(
        dataset_path=args.dataset,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        allow_overlap=args.allow_overlap,
    )
    print_evaluation_summary(summary)


if __name__ == "__main__":
    main()
