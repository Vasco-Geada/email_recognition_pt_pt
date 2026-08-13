# -*- coding: utf-8 -*-
"""Run the complete inference project using the latest persisted models.

Usage:
    python run_project.py dataset/imported_emails_anonymized.json
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from run_classification_models import (
    DEFAULT_MODEL_DIR,
    MODEL_NAMES,
    evaluate_loaded_classifier,
    load_all_models,
    load_training_metadata,
)
from run_project_evaluation import (
    PreparedDataset,
    run_classic_extraction,
    run_qa_module,
    write_json,
    write_summary_csv,
)
from models.utils import combine_text_fields, load_dataset, preprocess_texts
from preprocessing.preprocess import preprocessEmail


PROJECT_ROOT = Path(__file__).resolve().parent
TRAINED_MODELS_ROOT = PROJECT_ROOT / "trained_models"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "evaluation_results" / "project_runs"
DEFAULT_QA_MODEL = PROJECT_ROOT / "qa" / "models" / "bertimbau_qa_finetuned"
PRIMARY_INTENT_MODEL = "logistic_regression"

logger = logging.getLogger("run_project")


@dataclass
class InferenceDataset:
    raw_emails: List[Dict[str, Any]]
    processed_emails: List[Dict[str, Any]]
    texts: List[str]
    labels: List[Optional[str]]

# Resolve the directory containing the persisted classification models, preferring the canonical location but falling back to the most recently saved model bundle if necessary.
def resolve_classification_model_dir() -> Path:
    """Use the canonical directory, falling back to the latest saved bundle."""
    canonical_metadata = DEFAULT_MODEL_DIR / "training_metadata.json"
    if canonical_metadata.exists():
        return DEFAULT_MODEL_DIR

    candidates = list(TRAINED_MODELS_ROOT.glob("*/training_metadata.json"))
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime).parent
    raise FileNotFoundError(
        "Nao existem classificadores persistidos. Treina-os primeiro com: "
        "python run_classification_models.py train --dataset <dataset.json>"
    )

# Prepare the inference dataset by preprocessing the imported emails.
def prepare_inference_dataset(dataset_path: Path) -> InferenceDataset:
    """Preprocess an imported dataset without applying anonymization again."""
    raw_emails = load_dataset(str(dataset_path))
    processed_emails: List[Dict[str, Any]] = []
    texts: List[str] = []
    labels: List[Optional[str]] = []

    for index, email_item in enumerate(raw_emails):
        if not isinstance(email_item, dict):
            logger.warning("Email %s ignorado: formato invalido", index)
            continue
        email_data = dict(email_item)
        try:
            processed = preprocessEmail(email_data)
        except Exception as exc:
            logger.warning("Preprocessing falhou no email %s: %s", index, exc)
            processed = {
                "subject": email_data.get("subject", ""),
                "clean_body": email_data.get("body", ""),
            }

        for key, value in email_data.items():
            if key not in processed and key != "body":
                processed[key] = value
        processed["original_body"] = email_data.get("body", "")
        processed["original_index"] = index

        text = combine_text_fields(
            {
                "subject": processed.get("subject", ""),
                "body": processed.get("clean_body", ""),
            },
            subject_weight=1.0,
            body_weight=1.0,
        )
        if not text.strip():
            logger.warning("Email %s ignorado: assunto e corpo vazios", index)
            continue

        processed_emails.append(processed)
        texts.append(text)
        label = email_data.get("label")
        labels.append(str(label) if label else None)

    if not texts:
        raise ValueError("O dataset nao contem emails validos para processar.")
    texts = preprocess_texts(
        texts,
        remove_punctuation=False,
        lowercase=True,
    )
    return InferenceDataset(
        raw_emails=raw_emails,
        processed_emails=processed_emails,
        texts=texts,
        labels=labels,
    )


#Return a dictionary with predictions, probabilities, rows and metrics for a given model and dataset.
def _predict_model(
    model_name: str,
    classifier: Any,
    dataset: InferenceDataset,
) -> Dict[str, Any]:
    all_labelled = all(label is not None for label in dataset.labels)
    metrics = None
    if all_labelled:
        prepared = PreparedDataset(
            raw_emails=dataset.raw_emails,
            processed_emails=dataset.processed_emails,
            texts=dataset.texts,
            labels=[str(label) for label in dataset.labels],
        )
        metrics, predictions, probabilities = evaluate_loaded_classifier(
            model_name,
            classifier,
            prepared,
        )
    else:
        raw_predictions = classifier.predict(dataset.texts)
        raw_probabilities = classifier.predict_proba(dataset.texts)
        predictions = [str(value) for value in raw_predictions]
        probabilities = list(raw_probabilities)

    rows = []
    for index, (email_data, text, prediction, probability) in enumerate(
        zip(
            dataset.processed_emails,
            dataset.texts,
            predictions,
            probabilities,
        )
    ):
        row = {
            "email_index": email_data.get("original_index", index),
            "email_id": email_data.get("email_id"),
            "subject": email_data.get("subject", ""),
            "text": text,
            "predicted_label": prediction,
            "confidence": float(probability.get(prediction, 0.0)),
            "probabilities": probability,
        }
        if dataset.labels[index] is not None:
            row["true_label"] = dataset.labels[index]
        rows.append(row)
    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "rows": rows,
        "metrics": metrics,
    }

# Attach the primary intent prediction and consensus from multiple models to each email in the dataset.
def _attach_primary_intent(
    dataset: InferenceDataset,
    model_results: Dict[str, Dict[str, Any]],
) -> None:
    primary_predictions = model_results[PRIMARY_INTENT_MODEL]["predictions"]
    for index, email_data in enumerate(dataset.processed_emails):
        votes = Counter(
            model_results[model_name]["predictions"][index]
            for model_name in MODEL_NAMES
        )
        consensus = votes.most_common(1)[0][0]
        email_data["predicted_intent"] = primary_predictions[index]
        email_data["intent_consensus"] = consensus
        email_data["intent_predictions"] = {
            model_name: model_results[model_name]["predictions"][index]
            for model_name in MODEL_NAMES
        }

# Run the main project pipeline with the specified parameters.
def run_project(
    dataset_path: Path,
    model_dir: Optional[Path] = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    qa_model: Path = DEFAULT_QA_MODEL,
    include_classic: bool = True,
    include_qa: bool = True,
) -> Dict[str, Any]:
    """Run persisted classification, classic extraction and BERTimbau QA."""
    dataset_path = dataset_path.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {dataset_path}")

# Define variables
    selected_model_dir = model_dir or resolve_classification_model_dir()
    training_metadata = load_training_metadata(selected_model_dir)
    models = load_all_models(selected_model_dir)
    dataset = prepare_inference_dataset(dataset_path)

#Define output directories
    output_dir = output_root / dataset_path.stem
    classification_dir = output_dir / "classification"
    classification_dir.mkdir(parents=True, exist_ok=True)

# Run predictions for each model and save the results to JSON files, including metrics if available.
    model_results: Dict[str, Dict[str, Any]] = {}
    classification_metrics: Dict[str, Dict[str, Any]] = {}
    for model_name, classifier in models.items():
        logger.info("A executar %s sem treino", model_name)
        result = _predict_model(model_name, classifier, dataset)
        model_results[model_name] = result
        write_json(
            classification_dir / f"{model_name}_predictions.json",
            result["rows"],
        )
        if result["metrics"] is not None:
            classification_metrics[model_name] = result["metrics"]
            write_json(
                classification_dir / f"{model_name}_metrics.json",
                result["metrics"],
            )
    if classification_metrics:
        write_summary_csv(
            classification_dir / "summary.csv",
            classification_metrics,
        )
        
# Attach the primary intent predictions and consensus to the dataset, then save the updated dataset to a JSON file.
    _attach_primary_intent(dataset, model_results)
    write_json(output_dir / "emails_with_intent.json", dataset.processed_emails)

# Run classic extraction and QA modules if enabled, and summarize the results in a JSON file.
    reference_datetime = datetime.now()
    classic_count = None
    if include_classic:
        classic_count = run_classic_extraction(
            test_emails=dataset.processed_emails,
            output_dir=output_dir,
            reference_datetime=reference_datetime,
        )

# Run the QA module if enabled, ensuring that the specified QA model exists, and summarize the results in a JSON file.
    qa_count = None
    if include_qa:
        if not qa_model.exists():
            raise FileNotFoundError(f"Modelo QA nao encontrado: {qa_model}")
        qa_count = run_qa_module(
            test_emails=dataset.processed_emails,
            output_dir=output_dir,
            model_name=str(qa_model),
            reference_datetime=reference_datetime,
        )

# Summarize the project run, including counts of processed and anonymized emails, model details, and results from classic extraction and QA modules, then save the summary to a JSON file.
    anonymized_count = sum(
        isinstance(email_item.get("anonymization"), dict)
        for email_item in dataset.raw_emails
        if isinstance(email_item, dict)
    )
    summary = {
        "created_at": datetime.now().isoformat(),
        "dataset": str(dataset_path),
        "num_emails": len(dataset.texts),
        "anonymized_at_import_count": anonymized_count,
        "anonymization_applied_during_run": False,
        "classification": {
            "model_dir": str(selected_model_dir.resolve()),
            "training_dataset": training_metadata.get("training_dataset"),
            "models": list(MODEL_NAMES),
            "primary_intent_model": PRIMARY_INTENT_MODEL,
            "metrics": classification_metrics or None,
        },
        "classic_extraction": {
            "enabled": include_classic,
            "processed_emails": classic_count,
        },
        "qa": {
            "enabled": include_qa,
            "model": str(qa_model.resolve()) if include_qa else None,
            "processed_emails": qa_count,
        },
        "output_dir": str(output_dir.resolve()),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Executa classificacao, extracao classica e QA com os modelos "
            "persistidos. O unico argumento e o dataset."
        )
    )
    parser.add_argument("dataset", type=Path)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    summary = run_project(args.dataset)
    print("\n=== Projeto executado com modelos persistidos ===")
    print(f"Emails processados: {summary['num_emails']}")
    print(
        "Anonimizados na importacao: "
        f"{summary['anonymized_at_import_count']}"
    )
    print(f"Resultados: {summary['output_dir']}")


if __name__ == "__main__":
    main()
