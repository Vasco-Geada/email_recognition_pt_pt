# -*- coding: utf-8 -*-
"""
Decision Tree baseline for Portuguese email intent classification.

The classifier uses TF-IDF features over subject + body and a
sklearn.tree.DecisionTreeClassifier. It is intended as an interpretable
baseline for comparison with Naive Bayes, Logistic Regression and BERTimbau.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier


logger = logging.getLogger(__name__)


PredictionInput = Union[str, List[str]]


class DecisionTreeEmailClassifier:
    """
    TF-IDF + Decision Tree classifier for meeting-related email intents.

    Parameters mirror the requested baseline setup and keep the tree
    hyperparameters configurable for experimental comparison.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents=None,
            ngram_range=ngram_range,
            max_features=max_features,
        )
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        self.classes_: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str]) -> "DecisionTreeEmailClassifier":
        """
        Fit TF-IDF vectorizer and Decision Tree model.

        Raises:
            ValueError: If inputs are empty or lengths differ.
        """
        if not texts or not labels:
            raise ValueError("texts and labels cannot be empty")
        if len(texts) != len(labels):
            raise ValueError(
                f"texts and labels must have the same length: "
                f"{len(texts)} != {len(labels)}"
            )

        clean_texts = [self._safe_text(text) for text in texts]
        if not any(clean_texts):
            raise ValueError("all texts are empty after normalization")

        logger.info("Vectorizing %s texts with TF-IDF", len(clean_texts))
        x_train = self.vectorizer.fit_transform(clean_texts)

        logger.info("Training DecisionTreeClassifier")
        self.model.fit(x_train, labels)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        return self

    def predict(self, texts: PredictionInput) -> Union[str, List[str]]:
        """Predict the class for one text or a list of texts."""
        self._ensure_fitted()
        prepared, return_single = self._prepare_input(texts)
        x_values = self.vectorizer.transform(prepared)
        predictions = self.model.predict(x_values).tolist()
        return predictions[0] if return_single else predictions

    def predict_proba(
        self,
        texts: PredictionInput,
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """Return probabilities per class for one text or a list of texts."""
        self._ensure_fitted()
        prepared, return_single = self._prepare_input(texts)
        x_values = self.vectorizer.transform(prepared)
        proba = self.model.predict_proba(x_values)
        classes = [str(label) for label in self.model.classes_]

        results = [
            {cls: float(prob) for cls, prob in zip(classes, row)}
            for row in proba
        ]
        return results[0] if return_single else results

    def predict_with_confidence(self, text: str) -> Dict:
        """
        Predict a single email and return class, probabilities and confidence.
        """
        prediction = str(self.predict(text))
        probabilities = self.predict_proba(text)
        if not isinstance(probabilities, dict):
            raise RuntimeError("Expected single probability dictionary")

        return {
            "prediction": prediction,
            "probabilities": probabilities,
            "confidence": float(probabilities.get(prediction, 0.0)),
        }

    def evaluate(self, texts: List[str], labels: List[str], verbose: bool = True) -> Dict:
        """Evaluate the classifier with standard classification metrics."""
        self._ensure_fitted()
        predictions = self.predict(texts)
        if isinstance(predictions, str):
            predictions = [predictions]

        metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "precision_macro": precision_score(
                labels, predictions, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                labels, predictions, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
            "f1_weighted": f1_score(
                labels, predictions, average="weighted", zero_division=0
            ),
            "classification_report": classification_report(
                labels, predictions, zero_division=0
            ),
            "classification_report_dict": classification_report(
                labels, predictions, zero_division=0, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(
                labels, predictions, labels=list(self.model.classes_)
            ).tolist(),
            "classes": [str(label) for label in self.model.classes_],
            "predictions": predictions,
        }

        if verbose:
            logger.info("Accuracy: %.4f", metrics["accuracy"])
            logger.info("Macro F1: %.4f", metrics["f1_macro"])
            logger.info("Weighted F1: %.4f", metrics["f1_weighted"])
            logger.info("\n%s", metrics["classification_report"])
            logger.info("Confusion matrix:\n%s", metrics["confusion_matrix"])

        return metrics

    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Return the most important TF-IDF features according to the tree.

        Decision Trees expose global feature importances, not per-class
        coefficients like linear models.
        """
        self._ensure_fitted()
        feature_names = self.vectorizer.get_feature_names_out()
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        return [
            (str(feature_names[index]), float(importances[index]))
            for index in indices
            if importances[index] > 0
        ]

    def save(
        self,
        model_path: str = "models/decision_tree_model.joblib",
        vectorizer_path: str = "models/decision_tree_vectorizer.joblib",
    ) -> None:
        """Persist model and vectorizer with joblib."""
        self._ensure_fitted()
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(vectorizer_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        logger.info("Saved Decision Tree model to %s", model_path)
        logger.info("Saved TF-IDF vectorizer to %s", vectorizer_path)

    def load(
        self,
        model_path: str = "models/decision_tree_model.joblib",
        vectorizer_path: str = "models/decision_tree_vectorizer.joblib",
    ) -> "DecisionTreeEmailClassifier":
        """Load model and vectorizer from disk."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not Path(vectorizer_path).exists():
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        logger.info("Loaded Decision Tree model from %s", model_path)
        logger.info("Loaded TF-IDF vectorizer from %s", vectorizer_path)
        return self

    @staticmethod
    def _safe_text(value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _prepare_input(self, texts: PredictionInput) -> Tuple[List[str], bool]:
        if isinstance(texts, str):
            prepared = [self._safe_text(texts)]
            return_single = True
        else:
            prepared = [self._safe_text(text) for text in texts]
            return_single = False

        if not prepared or not any(prepared):
            raise ValueError("input text cannot be empty")
        return prepared, return_single

    def _ensure_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")

