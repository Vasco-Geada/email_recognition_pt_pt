"""TF-IDF + Logistic Regression classifier for PT-PT email intents."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


logger = logging.getLogger(__name__)
PredictionInput = Union[str, List[str]]


class LogisticRegressionEmailClassifier:
    """Classify email intent using TF-IDF unigram/bigram features."""

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        random_state: int = 42,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents=None,
            ngram_range=ngram_range,
            max_features=max_features,
        )
        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )
        self.classes_: Optional[np.ndarray] = None
        self.is_fitted = False

# Creates the TF-IDF vectorizer and fits the Logistic Regression model to the provided texts and labels.
    def fit(
        self,
        texts: List[str],
        labels: List[str],
    ) -> "LogisticRegressionEmailClassifier":
        if not texts or not labels:
            raise ValueError("texts and labels cannot be empty")
        if len(texts) != len(labels):
            raise ValueError(
                f"texts and labels must have the same length: "
                f"{len(texts)} != {len(labels)}"
            )

        prepared = [self._safe_text(text) for text in texts]
        if not any(prepared):
            raise ValueError("all texts are empty")

        x_train = self.vectorizer.fit_transform(prepared)
        self.model.fit(x_train, labels)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        return self
    
# Predicts the class labels for the given texts using the fitted Logistic Regression model.
    def predict(self, texts: PredictionInput) -> Union[str, List[str]]:
        self._ensure_fitted()
        prepared, return_single = self._prepare_input(texts)
        predictions = self.model.predict(
            self.vectorizer.transform(prepared)
        ).tolist()
        return predictions[0] if return_single else predictions


# Returns the prediction probabilities for each class using the fitted Logistic Regression model.
    def predict_proba(
        self,
        texts: PredictionInput,
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        self._ensure_fitted()
        prepared, return_single = self._prepare_input(texts)
        probabilities = self.model.predict_proba(
            self.vectorizer.transform(prepared)
        )
        classes = [str(label) for label in self.model.classes_]
        results = [
            {
                label: float(probability)
                for label, probability in zip(classes, row)
            }
            for row in probabilities
        ]
        return results[0] if return_single else results

    def save(
        self,
        model_path: str,
        vectorizer_path: str,
    ) -> None:
        self._ensure_fitted()
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(vectorizer_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        logger.info("Saved Logistic Regression model to %s", model_path)
        logger.info("Saved TF-IDF vectorizer to %s", vectorizer_path)

    def load(
        self,
        model_path: str,
        vectorizer_path: str,
    ) -> "LogisticRegressionEmailClassifier":
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not Path(vectorizer_path).exists():
            raise FileNotFoundError(
                f"Vectorizer file not found: {vectorizer_path}"
            )

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        return self

    def get_feature_importance(
        self,
        top_n: int = 20,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Return the features with the largest coefficient per class."""
        self._ensure_fitted()
        feature_names = np.asarray(self.vectorizer.get_feature_names_out())
        importance: Dict[str, List[Tuple[str, float]]] = {}
        for class_index, label in enumerate(self.model.classes_):
            indices = np.argsort(self.model.coef_[class_index])[-top_n:][::-1]
            importance[str(label)] = [
                (str(feature_names[index]), float(self.model.coef_[class_index, index]))
                for index in indices
            ]
        return importance

    @staticmethod
    def _safe_text(value: object) -> str:
        return "" if value is None else str(value).strip()

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
