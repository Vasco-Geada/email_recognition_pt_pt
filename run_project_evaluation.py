# -*- coding: utf-8 -*-
"""
Runner unificado para usar os principais modulos do projeto e comparar modelos.

Exemplo:
    python run_project_evaluation.py --dataset dataset/dataset.json
    python run_project_evaluation.py --dataset dataset/realistic_emails_v2.json --skip-anonymization

O script separa tres avaliacoes:
1. classificacao de intencao: Logistic Regression vs Naive Bayes vs Decision Tree;
2. extracao estruturada classica: regex/spaCy/heuristicas para argumentos;
3. QA com BERTimbau: perguntas sobre o email para os mesmos campos.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.decision_tree_classifier import DecisionTreeEmailClassifier
from models.logistic_regression_classifier import LogisticRegressionEmailClassifier
from models.naive_bayes_classifier import NaiveBayesEmailClassifier
from models.utils import combine_text_fields, load_dataset, preprocess_texts, validate_dataset
from preprocessing.anonymization import EmailAnonymizer
from preprocessing.preprocess import preprocessEmail
from preprocessing.temporal_normalization import (
    TemporalNormalizer,
    parse_datetime_value,
)
from preprocessing.trigger_extraction import TriggerExtractor

logger = logging.getLogger("run_project_evaluation")

BERTIMBAU_QA_CHECKPOINT = "pierreguillou/bert-base-cased-squad-v1.1-portuguese"


@dataclass
class PreparedDataset:
    raw_emails: List[Dict[str, Any]]
    processed_emails: List[Dict[str, Any]]
    texts: List[str]
    labels: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Executa o pipeline completo e compara a performance dos modelos."
    )
    parser.add_argument("--dataset", default="dataset/realistic_emails_v3.json", help="Dataset JSON.")
    parser.add_argument(
        "--output-dir",
        default="evaluation_results/full_pipeline",
        help="Diretorio onde guardar resultados.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Percentagem para teste.")
    parser.add_argument("--random-state", type=int, default=42, help="Seed reprodutivel.")
    parser.add_argument("--max-features", type=int, default=5000, help="Features TF-IDF.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Numero maximo de folds para cross-validation.")
    parser.add_argument("--skip-cv", action="store_true", help="Ignora cross-validation dos modelos de intencao.")
    parser.add_argument(
        "--skip-anonymization",
        action="store_true",
        help="Nao aplicar anonimizacao antes do preprocessing.",
    )
    parser.add_argument(
        "--skip-argument-extraction",
        action="store_true",
        help="Ignora a avaliacao de extracao estruturada classica.",
    )
    parser.add_argument(
        "--gold-annotations",
        default=None,
        help="Gold annotations para avaliar extracao classica e/ou QA.",
    )
    parser.add_argument(
        "--run-qa",
        action="store_true",
        help="Executa tambem o modulo QA sobre o conjunto de teste.",
    )
    parser.add_argument(
        "--qa-model",
        default="bertimbau-pt",
        help=(
            "Modelo QA a usar quando --run-qa estiver ativo. "
            "Aceita aliases BERTimbau ou um diretorio local fine-tuned."
        ),
    )
    parser.add_argument(
        "--qa-gold",
        default=None,
        help="Gold annotations especificas para avaliar o QA. Se omitido, usa --gold-annotations.",
    )
    parser.add_argument(
        "--reference-datetime",
        default=None,
        help="Data de referencia ISO para normalizacao temporal. Default: agora.",
    )
    return parser.parse_args()


def ensure_output_dir(path: str) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def prepare_dataset(dataset_path: str, use_anonymization: bool) -> PreparedDataset:
    logger.info("A validar dataset: %s", dataset_path)
    validation = validate_dataset(dataset_path)
    if validation.get("has_errors"):
        raise ValueError(f"Dataset invalido: {validation.get('errors')}")

    raw_emails = load_dataset(dataset_path)
    anonymizer = EmailAnonymizer(use_spacy=False) if use_anonymization else None

    processed_emails: List[Dict[str, Any]] = []
    texts: List[str] = []
    labels: List[str] = []

    for index, email in enumerate(raw_emails):
        label = email.get("label")
        if not label:
            logger.warning("Email %s ignorado: sem label.", index)
            continue

        working_email = dict(email)
        if anonymizer is not None and not working_email.get("anonymization"):
            working_email = anonymizer.anonymize_email(working_email)

        try:
            processed = preprocessEmail(working_email)
        except Exception as exc:
            logger.warning("Preprocessing falhou no email %s: %s", index, exc)
            processed = dict(working_email)
            processed["clean_body"] = working_email.get("body", "")

        for key, value in working_email.items():
            if key not in processed and key != "body":
                processed[key] = value
        processed["original_body"] = working_email.get("body", "")

        text = combine_text_fields(
            {
                "subject": processed.get("subject", working_email.get("subject", "")),
                "body": processed.get("clean_body", working_email.get("body", "")),
            },
            subject_weight=1.0,
            body_weight=1.0,
        )

        if not text.strip():
            logger.warning("Email %s ignorado: texto vazio apos preprocessing.", index)
            continue

        processed["original_index"] = index
        processed["label"] = label
        processed_emails.append(processed)
        texts.append(text)
        labels.append(str(label))

    texts = preprocess_texts(texts, remove_punctuation=False, lowercase=True)
    if len(set(labels)) < 2:
        raise ValueError("Sao necessarias pelo menos duas classes para avaliar modelos.")

    return PreparedDataset(raw_emails, processed_emails, texts, labels)


def build_models(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "logistic_regression": LogisticRegressionEmailClassifier(
            max_features=args.max_features,
            random_state=args.random_state,
        ),
        "naive_bayes": NaiveBayesEmailClassifier(
            max_features=args.max_features,
            random_state=args.random_state,
        ),
        "decision_tree": DecisionTreeEmailClassifier(
            max_features=args.max_features,
            random_state=args.random_state,
        ),
    }


def split_dataset(
    prepared: PreparedDataset,
    test_size: float,
    random_state: int,
) -> Tuple[List[str], List[str], List[str], List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    indices = list(range(len(prepared.texts)))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=prepared.labels,
    )

    def select(values: List[Any], selected: Iterable[int]) -> List[Any]:
        return [values[index] for index in selected]

    return (
        select(prepared.texts, train_idx),
        select(prepared.texts, test_idx),
        select(prepared.labels, train_idx),
        select(prepared.labels, test_idx),
        select(prepared.processed_emails, train_idx),
        select(prepared.processed_emails, test_idx),
    )


def parse_email_reference_datetime(
    email: Dict[str, Any],
    fallback: datetime,
) -> datetime:
    for key in (
        "sent_datetime",
        "sent_at",
        "email_date",
        "date",
        "Date",
        "created_at",
        "timestamp",
    ):
        value = email.get(key)
        if not value:
            continue
        parsed = parse_datetime_value(value)
        if parsed is not None:
            return parsed
        logger.warning("Data de envio invalida em %s: %s", key, value)
    return fallback


def build_cv_estimators(args: argparse.Namespace) -> Dict[str, Pipeline]:
    vectorizer_kwargs = {
        "lowercase": True,
        "strip_accents": None,
        "ngram_range": (1, 2),
        "max_features": args.max_features,
    }
    return {
        "logistic_regression": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**vectorizer_kwargs)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=args.random_state,
                    ),
                ),
            ]
        ),
        "naive_bayes": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**vectorizer_kwargs)),
                ("model", MultinomialNB(alpha=1.0)),
            ]
        ),
        "decision_tree": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**vectorizer_kwargs)),
                ("model", DecisionTreeClassifier(random_state=args.random_state)),
            ]
        ),
    }


def choose_cv_folds(labels: List[str], requested_folds: int) -> int:
    class_counts = {label: labels.count(label) for label in set(labels)}
    min_class_count = min(class_counts.values()) if class_counts else 0
    return max(0, min(requested_folds, min_class_count))


def run_cross_validation(
    texts: List[str],
    labels: List[str],
    args: argparse.Namespace,
) -> Dict[str, Dict[str, Any]]:
    folds = choose_cv_folds(labels, args.cv_folds)
    if args.skip_cv or folds < 2:
        logger.warning("Cross-validation ignorada: folds disponiveis=%s", folds)
        return {}

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=args.random_state,
    )
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
    }

    results = {}
    for model_name, estimator in build_cv_estimators(args).items():
        logger.info("Cross-validation %s (%s folds)", model_name, folds)
        started_at = time.perf_counter()
        scores = cross_validate(
            estimator,
            texts,
            labels,
            cv=cv,
            scoring=scoring,
            n_jobs=None,
            error_score="raise",
        )
        elapsed = time.perf_counter() - started_at
        model_cv = {"folds": folds, "seconds": elapsed}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            model_cv[f"{metric}_mean"] = float(values.mean())
            model_cv[f"{metric}_std"] = float(values.std())
            model_cv[f"{metric}_scores"] = [float(value) for value in values]
        results[model_name] = model_cv

    return results


def evaluate_classifier(
    model_name: str,
    model: Any,
    x_train: List[str],
    x_test: List[str],
    y_train: List[str],
    y_test: List[str],
) -> Dict[str, Any]:
    logger.info("A treinar %s", model_name)
    started_at = time.perf_counter()
    model.fit(x_train, y_train)
    train_seconds = time.perf_counter() - started_at

    started_at = time.perf_counter()
    predictions = model.predict(x_test)
    predict_seconds = time.perf_counter() - started_at
    probabilities = model.predict_proba(x_test)

    labels = sorted(set(y_train) | set(y_test) | set(predictions))
    metrics = {
        "model": model_name,
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=labels,
            zero_division=0,
            output_dict=True,
        ),
    }

    return {
        "model": model,
        "predictions": [str(prediction) for prediction in predictions],
        "probabilities": probabilities,
        "metrics": metrics,
    }


def run_downstream_modules(
    model_name: str,
    test_emails: List[Dict[str, Any]],
    test_texts: List[str],
    predictions: List[str],
    probabilities: List[Dict[str, float]],
    reference_datetime: datetime,
    run_arguments: bool,
) -> List[Dict[str, Any]]:
    trigger_extractor = TriggerExtractor(use_lemmatization=False)
    temporal_normalizer = TemporalNormalizer()
    argument_extractor = None

    if run_arguments:
        try:
            from preprocessing.argument_extraction import ArgumentExtractor

            argument_extractor = ArgumentExtractor()
        except Exception as exc:
            logger.warning(
                "Extracao de argumentos indisponivel para %s: %s",
                model_name,
                exc,
            )

    rows: List[Dict[str, Any]] = []
    for index, (email, text, prediction, proba) in enumerate(
        zip(test_emails, test_texts, predictions, probabilities)
    ):
        body = email.get("clean_body") or email.get("body") or text
        subject = email.get("subject", "")
        email_reference_datetime = parse_email_reference_datetime(email, reference_datetime)

        try:
            trigger = trigger_extractor.extract_trigger(body, prediction)
        except Exception as exc:
            logger.warning("Trigger extraction falhou em %s/%s: %s", model_name, index, exc)
            trigger = None

        arguments = None
        temporal_normalized: List[Dict[str, Any]] = []

        if argument_extractor is not None:
            try:
                arguments = argument_extractor.extract_with_context(
                    email_body=body,
                    email_subject=subject,
                    predicted_intent=prediction,
                    trigger=(trigger or {}).get("trigger", ""),
                )
                time_spans = (
                    arguments.get("extracted_arguments", {})
                    .get("time_expressions", [])
                )
                for span in time_spans:
                    normalized = temporal_normalizer.normalize(
                        span.get("text", ""),
                        reference_datetime=email_reference_datetime,
                    )
                    temporal_normalized.append(normalized.to_dict())
            except Exception as exc:
                logger.warning("Argument extraction falhou em %s/%s: %s", model_name, index, exc)

        confidence = float(proba.get(prediction, 0.0)) if isinstance(proba, dict) else 0.0
        rows.append(
            {
                "email_index": email.get("original_index", index),
                "subject": subject,
                "text": text,
                "predicted_label": prediction,
                "confidence": confidence,
                "probabilities": proba,
                "trigger": trigger,
                "arguments": arguments,
                "normalized_temporals": temporal_normalized,
                "reference_datetime": email_reference_datetime.isoformat(),
            }
        )

    return rows


def run_classic_extraction(
    test_emails: List[Dict[str, Any]],
    output_dir: Path,
    reference_datetime: datetime,
) -> Optional[int]:
    """
    Run the classic structured extraction pipeline independently of intent models.

    This covers regex/spaCy/heuristic extractors for participants, time,
    location and topic, plus temporal normalization for extracted times.
    """
    try:
        from preprocessing.argument_extraction import ArgumentExtractor

        extractor = ArgumentExtractor()
        normalizer = TemporalNormalizer()
        results = []

        for index, email in enumerate(test_emails):
            body = email.get("clean_body") or email.get("body") or ""
            subject = email.get("subject", "")
            email_reference_datetime = parse_email_reference_datetime(email, reference_datetime)
            extracted = extractor.extract_with_context(
                email_body=body,
                email_subject=subject,
                predicted_intent=(
                    email.get("predicted_intent")
                    or email.get("label", "")
                ),
                trigger="",
            )
            arguments = extracted.get("extracted_arguments", {})

            qa_like_results = {
                "participants": _classic_category_result(
                    arguments.get("participants", []),
                    "Quem participa na reuniao?",
                ),
                "time": _classic_category_result(
                    arguments.get("time_expressions", []),
                    "Quando e a reuniao?",
                ),
                "location": _classic_category_result(
                    _normalize_location_spans(arguments.get("locations", [])),
                    "Onde e a reuniao?",
                ),
                "topic": _classic_category_result(
                    arguments.get("topics", []),
                    "Qual e o topico da reuniao?",
                ),
            }

            normalized_temporals = []
            for span in arguments.get("time_expressions", []):
                normalized = normalizer.normalize(
                    span.get("text", ""),
                    reference_datetime=email_reference_datetime,
                )
                normalized_temporals.append(normalized.to_dict())
            if normalized_temporals:
                qa_like_results["time"]["normalized"] = normalized_temporals[0]

            results.append(
                {
                    "email_id": email.get("original_index", index),
                    "email_text": body,
                    "subject": subject,
                    "qa_results": qa_like_results,
                    "classic_arguments": arguments,
                    "normalized_temporals": normalized_temporals,
                    "reference_datetime": email_reference_datetime.isoformat(),
                    "metadata": {
                        "method": "regex_spacy_heuristics",
                        "source": "preprocessing.argument_extraction.ArgumentExtractor",
                        "sent_datetime": email.get("sent_datetime"),
                        "reference_datetime": email_reference_datetime.isoformat(),
                    },
                }
            )

        classic_dir = output_dir / "classic_extraction"
        classic_dir.mkdir(parents=True, exist_ok=True)
        write_json(classic_dir / "classic_results.json", results)
        write_classic_results_csv(classic_dir / "classic_results.csv", results)
        return len(results)
    except Exception as exc:
        logger.warning("Extracao estruturada classica nao foi executada: %s", exc)
        return None


def _classic_category_result(spans: List[Dict[str, Any]], question: str) -> Dict[str, Any]:
    texts = [str(span.get("text", "")).strip() for span in spans if str(span.get("text", "")).strip()]
    confidences = [
        float(span.get("confidence", 0.0))
        for span in spans
        if str(span.get("text", "")).strip()
    ]
    return {
        "answer": "; ".join(texts) if texts else None,
        "confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "question": question,
        "valid": bool(texts),
    }


def _normalize_location_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_spans: List[Dict[str, Any]] = []
    for span in spans:
        text = str(span.get("text", "")).strip()
        normalized = _normalize_location_value(text)
        if not normalized:
            continue
        updated = dict(span)
        updated["text"] = normalized
        normalized_spans.append(updated)
    return normalized_spans


def _normalize_location_value(value: str) -> Optional[str]:
    text = re.sub(r"^\s*(?:em|no|na|por|via)\s+", "", str(value or "").strip(), flags=re.IGNORECASE)
    if not text:
        return None
    remote_match = re.search(r"\b(zoom|teams|microsoft\s+teams|google\s+meet|meet|online|remot[ao])\b", text, re.IGNORECASE)
    if remote_match:
        return f"remoto - {_normalize_remote_platform(remote_match.group(0))}"
    room_department_match = re.search(
        r"\bsala\s+de\s+reuni(?:ao|oes|ões|Ãµes)\s+do\s+departamento\b",
        text,
        re.IGNORECASE,
    )
    if room_department_match:
        return f"presencial - {room_department_match.group(0).strip()}"
    if re.search(
        r"\bsala(?:\s+de\s+reuni(?:ao|oes|ões))?(?:\s+[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)?)?(?:\s+do\s+departamento)?\b",
        text,
        re.IGNORECASE,
    ):
        return f"presencial - {text}"
    office_match = re.search(
        r"\bgabinete(?:\s+(?:da\s+direcao|da\s+direção|de\s+\w+|\d+(?:[.\-]\d+)?))?\b",
        text,
        re.IGNORECASE,
    )
    if office_match:
        return f"presencial - {office_match.group(0).strip()}"
    presential_match = re.search(
        r"\b(secretaria|biblioteca|laborat(?:orio|ório)(?:\s+de\s+\w+)?|audit(?:orio|ório)(?:\s+\w+)?|campus(?:\s+\w+)?)\b",
        text,
        re.IGNORECASE,
    )
    if presential_match:
        return f"presencial - {presential_match.group(0).strip()}"
    return None


def _normalize_remote_platform(platform: str) -> str:
    text = str(platform or "").strip().lower()
    if "team" in text:
        return "teams"
    if "meet" in text:
        return "google meet"
    if "zoom" in text:
        return "zoom"
    return "zoom"


def write_classic_results_csv(path: Path, results: List[Dict[str, Any]]) -> None:
    columns = ["email_id", "subject", "participants", "time", "location", "topic"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for result in results:
            qa_results = result.get("qa_results", {})
            writer.writerow(
                {
                    "email_id": result.get("email_id"),
                    "subject": result.get("subject", ""),
                    "participants": (qa_results.get("participants") or {}).get("answer") or "",
                    "time": (qa_results.get("time") or {}).get("answer") or "",
                    "location": (qa_results.get("location") or {}).get("answer") or "",
                    "topic": (qa_results.get("topic") or {}).get("answer") or "",
                }
            )


def compare_extraction_methods(
    classic_metrics: Optional[Dict[str, Any]],
    qa_metrics: Optional[Dict[str, Any]],
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Compare classic extraction against BERTimbau QA on shared metrics."""
    if not classic_metrics or not qa_metrics:
        return None

    comparison = {
        "methods": {
            "classic": "regex/spaCy/heuristics",
            "qa": "BERTimbau QA",
        },
        "rows": [],
    }

    scopes = [("global", "all")]
    categories = sorted(
        set(classic_metrics.get("per_category", {}).keys())
        | set(qa_metrics.get("per_category", {}).keys())
    )
    scopes.extend(("category", category) for category in categories)

    for scope, category in scopes:
        classic_item = _metric_scope(classic_metrics, category)
        qa_item = _metric_scope(qa_metrics, category)
        row = {
            "scope": scope,
            "category": category,
            "classic_em": classic_item["em"],
            "classic_f1": classic_item["f1"],
            "qa_em": qa_item["em"],
            "qa_f1": qa_item["f1"],
            "delta_em_qa_minus_classic": qa_item["em"] - classic_item["em"],
            "delta_f1_qa_minus_classic": qa_item["f1"] - classic_item["f1"],
            "winner_f1": _winner(classic_item["f1"], qa_item["f1"]),
        }
        comparison["rows"].append(row)

    comparison_dir = output_dir / "method_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    write_json(comparison_dir / "classic_vs_bertimbau_qa.json", comparison)
    write_method_comparison_csv(
        comparison_dir / "classic_vs_bertimbau_qa.csv",
        comparison["rows"],
    )
    return comparison


def _metric_scope(metrics: Dict[str, Any], category: str) -> Dict[str, float]:
    if category == "all":
        return {
            "em": float(metrics.get("exact_match_score", 0.0)),
            "f1": float(metrics.get("mean_f1_score", 0.0)),
        }

    item = metrics.get("per_category", {}).get(category, {})
    return {
        "em": float(item.get("exact_match", 0.0)),
        "f1": float(item.get("f1_score", 0.0)),
    }


def _winner(classic_f1: float, qa_f1: float) -> str:
    if qa_f1 > classic_f1:
        return "bertimbau_qa"
    if classic_f1 > qa_f1:
        return "classic"
    return "tie"


def write_method_comparison_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    columns = [
        "scope",
        "category",
        "classic_em",
        "classic_f1",
        "qa_em",
        "qa_f1",
        "delta_em_qa_minus_classic",
        "delta_f1_qa_minus_classic",
        "winner_f1",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_classification_predictions(
    test_emails: List[Dict[str, Any]],
    test_texts: List[str],
    y_test: List[str],
    predictions: List[str],
    probabilities: List[Dict[str, float]],
) -> List[Dict[str, Any]]:
    rows = []
    for index, (email, text, true_label, prediction, proba) in enumerate(
        zip(test_emails, test_texts, y_test, predictions, probabilities)
    ):
        rows.append(
            {
                "email_index": email.get("original_index", index),
                "subject": email.get("subject", ""),
                "text": text,
                "true_label": true_label,
                "predicted_label": prediction,
                "confidence": float(proba.get(prediction, 0.0)) if isinstance(proba, dict) else 0.0,
                "probabilities": proba,
            }
        )
    return rows


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=str)


def write_summary_csv(path: Path, metrics: Dict[str, Dict[str, Any]]) -> None:
    columns = [
        "model",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "cv_f1_macro_mean",
        "cv_f1_macro_std",
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "train_seconds",
        "predict_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for model_name, model_metrics in metrics.items():
            row = {column: model_metrics.get(column) for column in columns}
            row["model"] = model_name
            cv = model_metrics.get("cross_validation", {})
            row["cv_f1_macro_mean"] = cv.get("f1_macro_mean")
            row["cv_f1_macro_std"] = cv.get("f1_macro_std")
            row["cv_accuracy_mean"] = cv.get("accuracy_mean")
            row["cv_accuracy_std"] = cv.get("accuracy_std")
            writer.writerow(row)


def write_confusion_matrix_csv(path: Path, labels: List[str], matrix: List[List[int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def build_error_analysis(
    predictions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    errors = [
        item for item in predictions
        if item.get("true_label") != item.get("predicted_label")
    ]
    by_true_label: Dict[str, int] = {}
    by_predicted_label: Dict[str, int] = {}
    pairs: Dict[str, int] = {}

    for error in errors:
        true_label = str(error.get("true_label"))
        predicted_label = str(error.get("predicted_label"))
        by_true_label[true_label] = by_true_label.get(true_label, 0) + 1
        by_predicted_label[predicted_label] = by_predicted_label.get(predicted_label, 0) + 1
        pair_key = f"{true_label} -> {predicted_label}"
        pairs[pair_key] = pairs.get(pair_key, 0) + 1

    return {
        "total": len(predictions),
        "num_errors": len(errors),
        "error_rate": len(errors) / len(predictions) if predictions else 0.0,
        "by_true_label": by_true_label,
        "by_predicted_label": by_predicted_label,
        "confusion_pairs": pairs,
        "errors": errors,
    }


def write_error_analysis_csv(path: Path, errors: List[Dict[str, Any]]) -> None:
    columns = [
        "email_index",
        "subject",
        "true_label",
        "predicted_label",
        "confidence",
        "text",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for error in errors:
            writer.writerow({column: error.get(column) for column in columns})


def run_qa_module(
    test_emails: List[Dict[str, Any]],
    output_dir: Path,
    model_name: str,
    reference_datetime: datetime,
) -> Optional[int]:
    """Run the QA module once over the test set, when explicitly requested."""
    try:
        import torch

        qa_dir = PROJECT_ROOT / "qa"
        if str(qa_dir) not in sys.path:
            sys.path.insert(0, str(qa_dir))

        from qa_pipeline import QAPipeline

        qa_emails = []
        for index, email in enumerate(test_emails):
            qa_email = {
                "id": email.get("original_index", index),
                "subject": email.get("subject", ""),
                "text": email.get("clean_body") or email.get("body") or email.get("original_body") or "",
            }
            for key, value in email.items():
                if key not in qa_email and key not in {"clean_body", "sentences", "tokens"}:
                    qa_email[key] = value
            qa_emails.append(qa_email)

        pipeline = QAPipeline(
            model_name=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            confidence_threshold=0.3,
            verbose=True,
            reference_datetime=reference_datetime,
        )
        results = pipeline.process_batch(qa_emails, show_progress=True)
        pipeline.save_results(results, str(output_dir / "qa"), formats=["json", "csv"])
        return len(results)
    except Exception as exc:
        logger.warning("Modulo QA nao foi executado: %s", exc)
        return None


def evaluate_qa_module(
    gold_file: str,
    qa_predictions_file: Path,
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Evaluate QA outputs against gold annotations, if requested."""
    try:
        qa_dir = PROJECT_ROOT / "qa"
        if str(qa_dir) not in sys.path:
            sys.path.insert(0, str(qa_dir))

        from qa_evaluator import ProjectQAEvaluation

        evaluator = ProjectQAEvaluation.evaluate_files(
            gold_file=gold_file,
            predictions_file=str(qa_predictions_file),
            output_dir=str(output_dir / "qa_evaluation"),
            verbose=True,
        )
        return evaluator.aggregated_metrics.to_dict() if evaluator.aggregated_metrics else None
    except Exception as exc:
        logger.warning("Avaliacao QA nao foi executada: %s", exc)
        return None


def evaluate_extraction_outputs(
    gold_file: str,
    predictions_file: Path,
    output_dir: Path,
    evaluation_subdir: str,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """Evaluate structured extraction outputs with the shared QA-style evaluator."""
    try:
        qa_dir = PROJECT_ROOT / "qa"
        if str(qa_dir) not in sys.path:
            sys.path.insert(0, str(qa_dir))

        from qa_evaluator import ProjectQAEvaluation

        evaluator = ProjectQAEvaluation.evaluate_files(
            gold_file=gold_file,
            predictions_file=str(predictions_file),
            output_dir=str(output_dir / evaluation_subdir),
            verbose=verbose,
        )
        return evaluator.aggregated_metrics.to_dict() if evaluator.aggregated_metrics else None
    except Exception as exc:
        logger.warning("Avaliacao %s nao foi executada: %s", evaluation_subdir, exc)
        return None


def print_summary(
    metrics: Dict[str, Dict[str, Any]],
    classic_metrics: Optional[Dict[str, Any]] = None,
    classic_processed_count: Optional[int] = None,
    qa_model: Optional[str] = None,
    qa_metrics: Optional[Dict[str, Any]] = None,
    qa_processed_count: Optional[int] = None,
    method_comparison: Optional[Dict[str, Any]] = None,
) -> None:
    print("\n1. Classificacao de intencao")
    print("-" * 86)
    print(
        f"{'Modelo':<22} {'Accuracy':>10} {'F1 macro':>10} "
        f"{'CV F1':>10} {'CV std':>8} {'Treino(s)':>10} {'Pred(s)':>10}"
    )
    print("-" * 86)
    for model_name, item in sorted(
        metrics.items(),
        key=lambda pair: pair[1].get("f1_macro", 0.0),
        reverse=True,
    ):
        cv = item.get("cross_validation", {})
        print(
            f"{model_name:<22} "
            f"{item['accuracy']:>10.4f} "
            f"{item['f1_macro']:>10.4f} "
            f"{cv.get('f1_macro_mean', 0.0):>10.4f} "
            f"{cv.get('f1_macro_std', 0.0):>8.4f} "
            f"{item['train_seconds']:>10.3f} "
            f"{item['predict_seconds']:>10.3f}"
        )

    print("\n2. Extracao estruturada classica")
    print("-" * 86)
    print("Metodo: regex/spaCy/heuristicas")
    if classic_processed_count is not None:
        print(f"Emails processados: {classic_processed_count}")
    if classic_metrics:
        _print_extraction_metrics(classic_metrics)
    else:
        print("Metricas nao disponiveis. Use --gold-annotations para avaliar contra gold.")

    print("\n3. QA com BERTimbau")
    print("-" * 86)
    if qa_model:
        print(f"{'QA model':<22} {qa_model}")
        print(f"{'QA checkpoint':<22} {BERTIMBAU_QA_CHECKPOINT}")
        if qa_processed_count is not None:
            print(f"{'QA emails':<22} {qa_processed_count}")
    else:
        print("QA nao executado. Use --run-qa para ativar.")

    if qa_metrics:
        _print_extraction_metrics(qa_metrics)
    elif qa_model:
        print("Metricas QA nao disponiveis. Use --qa-gold ou --gold-annotations.")

    print("\n4. Comparacao: ArgumentExtractor classico vs BERTimbau QA")
    print("-" * 86)
    if method_comparison:
        print(
            f"{'Categoria':<18} {'Classic F1':>12} {'QA F1':>12} "
            f"{'Delta QA':>12} {'Vencedor':>16}"
        )
        print("-" * 86)
        for row in method_comparison.get("rows", []):
            print(
                f"{row['category']:<18} "
                f"{row['classic_f1']:>12.4f} "
                f"{row['qa_f1']:>12.4f} "
                f"{row['delta_f1_qa_minus_classic']:>12.4f} "
                f"{row['winner_f1']:>16}"
            )
    else:
        print("Comparacao indisponivel. Execute extracao classica, --run-qa e --gold-annotations.")


def _print_extraction_metrics(metrics: Dict[str, Any]) -> None:
    print(
        f"{'Categoria':<18} {'N':>5} {'EM':>10} {'Precision':>10} "
        f"{'Recall':>10} {'F1':>10} {'Conf.':>10}"
    )
    print("-" * 86)
    print(
        f"{'global':<18} "
        f"{metrics.get('total_examples', 0):>5} "
        f"{metrics.get('exact_match_score', 0.0):>10.4f} "
        f"{metrics.get('mean_precision', 0.0):>10.4f} "
        f"{metrics.get('mean_recall', 0.0):>10.4f} "
        f"{metrics.get('mean_f1_score', 0.0):>10.4f} "
        f"{metrics.get('mean_confidence', 0.0):>10.4f}"
    )
    for category, item in metrics.get("per_category", {}).items():
        print(
            f"{category:<18} "
            f"{item.get('count', 0):>5} "
            f"{item.get('exact_match', 0.0):>10.4f} "
            f"{item.get('precision', 0.0):>10.4f} "
            f"{item.get('recall', 0.0):>10.4f} "
            f"{item.get('f1_score', 0.0):>10.4f} "
            f"{item.get('mean_confidence', 0.0):>10.4f}"
        )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    output_dir = ensure_output_dir(args.output_dir)
    reference_datetime = (
        datetime.fromisoformat(args.reference_datetime)
        if args.reference_datetime
        else datetime.now()
    )

    prepared = prepare_dataset(
        dataset_path=args.dataset,
        use_anonymization=not args.skip_anonymization,
    )
    x_train, x_test, y_train, y_test, _, test_emails = split_dataset(
        prepared,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    logger.info(
        "Dataset preparado: %s exemplos treino, %s exemplos teste, %s classes.",
        len(x_train),
        len(x_test),
        len(set(prepared.labels)),
    )

    cv_results = run_cross_validation(prepared.texts, prepared.labels, args)
    all_metrics: Dict[str, Dict[str, Any]] = {}
    all_predictions: Dict[str, List[Dict[str, Any]]] = {}
    gold_file = args.gold_annotations or args.qa_gold
    classification_dir = output_dir / "classification"
    classification_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in build_models(args).items():
        result = evaluate_classifier(model_name, model, x_train, x_test, y_train, y_test)
        if model_name in cv_results:
            result["metrics"]["cross_validation"] = cv_results[model_name]
        all_metrics[model_name] = result["metrics"]
        model_predictions = build_classification_predictions(
            test_emails=test_emails,
            test_texts=x_test,
            y_test=y_test,
            predictions=result["predictions"],
            probabilities=result["probabilities"],
        )
        all_predictions[model_name] = model_predictions

        write_json(classification_dir / f"{model_name}_metrics.json", result["metrics"])
        write_json(classification_dir / f"{model_name}_predictions.json", model_predictions)
        write_confusion_matrix_csv(
            classification_dir / f"{model_name}_confusion_matrix.csv",
            result["metrics"]["labels"],
            result["metrics"]["confusion_matrix"],
        )
        error_analysis = build_error_analysis(model_predictions)
        write_json(classification_dir / f"{model_name}_error_analysis.json", error_analysis)
        write_error_analysis_csv(
            classification_dir / f"{model_name}_errors.csv",
            error_analysis["errors"],
        )

        # Backward-compatible copies at the run root for older notebooks/views.
        write_json(output_dir / f"{model_name}_metrics.json", result["metrics"])
        write_json(output_dir / f"{model_name}_predictions.json", model_predictions)

    write_summary_csv(classification_dir / "summary.csv", all_metrics)

    classic_processed_count = None
    classic_metrics = None
    if not args.skip_argument_extraction:
        classic_processed_count = run_classic_extraction(
            test_emails=test_emails,
            output_dir=output_dir,
            reference_datetime=reference_datetime,
        )
        if gold_file:
            classic_metrics = evaluate_extraction_outputs(
                gold_file=gold_file,
                predictions_file=output_dir / "classic_extraction" / "classic_results.json",
                output_dir=output_dir,
                evaluation_subdir="classic_extraction/evaluation",
                verbose=False,
            )

    summary = {
        "created_at": datetime.now().isoformat(),
        "dataset": args.dataset,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "reference_datetime": reference_datetime.isoformat(),
        "num_examples": len(prepared.texts),
        "num_train": len(x_train),
        "num_test": len(x_test),
        "gold_annotations": gold_file,
        "evaluations": {
            "classification": {
                "enabled": True,
                "models": list(all_metrics.keys()),
                "cross_validation_enabled": not args.skip_cv,
                "cv_folds_requested": args.cv_folds,
                "output_dir": str(classification_dir),
            },
            "classic_extraction": {
                "enabled": not args.skip_argument_extraction,
                "method": "regex/spaCy/heuristics",
                "processed_emails": classic_processed_count,
                "output_dir": str(output_dir / "classic_extraction"),
            },
            "qa": {
                "enabled": args.run_qa,
                "model": args.qa_model if args.run_qa else None,
                "checkpoint": BERTIMBAU_QA_CHECKPOINT if args.run_qa else None,
                "gold_file": (args.qa_gold or args.gold_annotations) if args.run_qa else None,
            },
        },
        "qa": {
            "enabled": args.run_qa,
            "model": args.qa_model if args.run_qa else None,
            "checkpoint": BERTIMBAU_QA_CHECKPOINT if args.run_qa else None,
            "gold_file": (args.qa_gold or args.gold_annotations) if args.run_qa else None,
        },
        "metrics": all_metrics,
    }
    write_summary_csv(output_dir / "summary.csv", all_metrics)

    qa_processed_count = None
    qa_metrics = None
    if args.run_qa:
        qa_processed_count = run_qa_module(
            test_emails,
            output_dir,
            args.qa_model,
            reference_datetime=reference_datetime,
        )
        qa_gold_file = args.qa_gold or args.gold_annotations
        if qa_gold_file:
            qa_metrics = evaluate_extraction_outputs(
                gold_file=qa_gold_file,
                predictions_file=output_dir / "qa" / "qa_results.json",
                output_dir=output_dir,
                evaluation_subdir="qa_evaluation",
                verbose=False,
            )

    method_comparison = compare_extraction_methods(
        classic_metrics=classic_metrics,
        qa_metrics=qa_metrics,
        output_dir=output_dir,
    )
    summary["classic_extraction_metrics"] = classic_metrics
    summary["qa_metrics"] = qa_metrics
    summary["method_comparison"] = method_comparison
    write_json(output_dir / "summary.json", summary)

    print_summary(
        all_metrics,
        classic_metrics=classic_metrics,
        classic_processed_count=classic_processed_count,
        qa_model=args.qa_model if args.run_qa else None,
        qa_metrics=qa_metrics,
        qa_processed_count=qa_processed_count,
        method_comparison=method_comparison,
    )
    print(f"\nResultados guardados em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
