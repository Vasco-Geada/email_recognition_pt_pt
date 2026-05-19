# -*- coding: utf-8 -*-
"""
Módulo de Classificação de Emails usando Naive Bayes Multinomial.

Este módulo fornece uma implementação completa de um classificador de emails
usando Naive Bayes com TF-IDF para português europeu.

Classes:
    NaiveBayesEmailClassifier: Classificador principal com TF-IDF e Naive Bayes.

Exemplo:
    >>> from models.naive_bayes_classifier import NaiveBayesEmailClassifier
    >>> clf = NaiveBayesEmailClassifier(max_features=5000, ngram_range=(1, 2))
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    >>> confidence = clf.predict_proba(X_test)
"""

import logging
import numpy as np
import joblib
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NaiveBayesEmailClassifier:
    """
    Classificador de emails usando Naive Bayes Multinomial com TF-IDF.
    
    Este classificador combina TF-IDF (Term Frequency-Inverse Document Frequency)
    com Naive Bayes Multinomial para classificação de emails em português europeu.
    
    Attributes:
        vectorizer (TfidfVectorizer): Extrator de features TF-IDF.
        model (MultinomialNB): Modelo Naive Bayes.
        classes_ (np.ndarray): Classes de classificação aprendidas.
        is_fitted (bool): Indica se o modelo foi treinado.
        max_features (int): Número máximo de features TF-IDF.
        ngram_range (Tuple[int, int]): Intervalo de n-gramas.
        alpha (float): Parâmetro de suavização Laplace.
    """
    
    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        alpha: float = 1.0,
        use_idf: bool = True,
        use_stopwords: bool = False,
        min_df: int = 1,
        max_df: float = 1.0,
        random_state: int = 42
    ) -> None:
        """
        Inicializa o classificador Naive Bayes.
        
        Args:
            max_features: Número máximo de features a extrair. Default: 5000.
            ngram_range: Intervalo de n-gramas (min_n, max_n). Default: (1, 2).
            alpha: Parâmetro de suavização Laplace. Default: 1.0.
            use_idf: Se True, usa TF-IDF; se False, usa TF. Default: True.
            use_stopwords: Se True, remove stopwords em português. Default: False.
            min_df: Frequência mínima de documento. Default: 1.
            max_df: Proporção máxima de documentos. Default: 1.0.
            random_state: Seed para reprodutibilidade. Default: 42.
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.alpha = alpha
        self.use_idf = use_idf
        self.use_stopwords = use_stopwords
        self.min_df = min_df
        self.max_df = max_df
        self.random_state = random_state
        
        self.classes_ = None
        self.is_fitted = False
        
        # Configurar stopwords português
        stopwords_pt = None
        if use_stopwords:
            stopwords_pt = {
                'a', 'à', 'o', 'os', 'as', 'um', 'uma', 'uns', 'umas',
                'de', 'do', 'da', 'dos', 'das', 'e', 'ou', 'é', 'são',
                'em', 'com', 'para', 'por', 'que', 'se', 'não', 'sim',
                'este', 'esse', 'aquele', 'este', 'esse', 'aquele',
                'meu', 'teu', 'seu', 'nosso', 'vosso'
            }
        
        # Inicializar TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            use_idf=use_idf,
            lowercase=True,
            stop_words=stopwords_pt,
            min_df=min_df,
            max_df=max_df,
            encoding='utf-8'
        )
        
        # Inicializar Naive Bayes
        self.model = MultinomialNB(alpha=alpha)
        
        logger.info(
            f"NaiveBayesEmailClassifier inicializado com "
            f"max_features={max_features}, "
            f"ngram_range={ngram_range}, "
            f"alpha={alpha}"
        )
    
    def fit(
        self,
        X: List[str],
        y: List[str]
    ) -> 'NaiveBayesEmailClassifier':
        """
        Treina o classificador.
        
        Args:
            X: Lista de textos de treino.
            y: Lista de labels de treino.
        
        Returns:
            self: Retorna a instância do classificador (para encadeamento).
        
        Raises:
            ValueError: Se X ou y estiverem vazios.
            TypeError: Se X ou y não forem do tipo esperado.
        """
        if not X or not y:
            raise ValueError("X e y não podem estar vazios")
        
        if len(X) != len(y):
            raise ValueError(
                f"X e y devem ter o mesmo tamanho. "
                f"Recebido X={len(X)}, y={len(y)}"
            )
        
        logger.info(f"Iniciando treino com {len(X)} exemplos")
        
        try:
            # Converter para lowercase e garantir UTF-8
            X_processed = [
                str(text).lower() if isinstance(text, str) else str(text).lower()
                for text in X
            ]
            
            # Vetorizar textos
            logger.info("Vetorizando textos com TF-IDF...")
            X_vectorized = self.vectorizer.fit_transform(X_processed)
            logger.info(f"Matriz TF-IDF criada com shape: {X_vectorized.shape}")
            
            # Treinar modelo
            logger.info("Treinando modelo Naive Bayes...")
            self.model.fit(X_vectorized, y)
            
            # Armazenar classes
            self.classes_ = self.model.classes_
            self.is_fitted = True
            
            logger.info(
                f"Treino concluído. Classes: {list(self.classes_)}"
            )
            
            return self
            
        except Exception as e:
            logger.error(f"Erro durante o treino: {str(e)}")
            raise
    
    def predict(self, X: Union[List[str], str]) -> Union[List[str], str]:
        """
        Realiza predições sobre novos textos.
        
        Args:
            X: Texto(s) a classificar. Pode ser string ou lista de strings.
        
        Returns:
            Predição(ões). String se X é string, lista se X é lista.
        
        Raises:
            RuntimeError: Se o modelo não foi treinado.
            ValueError: Se X está vazio.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Modelo não foi treinado. "
                "Use fit() antes de predict()"
            )
        
        # Lidar com string individual
        if isinstance(X, str):
            X = [X]
            return_single = True
        else:
            return_single = False
        
        if not X:
            raise ValueError("X não pode estar vazio")
        
        try:
            # Processar textos
            X_processed = [
                str(text).lower() if isinstance(text, str) else str(text).lower()
                for text in X
            ]
            
            # Vetorizar
            X_vectorized = self.vectorizer.transform(X_processed)
            
            # Predizer
            predictions = self.model.predict(X_vectorized)
            
            if return_single:
                return predictions[0]
            return list(predictions)
            
        except Exception as e:
            logger.error(f"Erro durante predição: {str(e)}")
            raise
    
    def predict_proba(
        self,
        X: Union[List[str], str]
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Retorna probabilidades de cada classe.
        
        Args:
            X: Texto(s) a classificar.
        
        Returns:
            Dicionário(s) com probabilidades por classe.
        
        Raises:
            RuntimeError: Se o modelo não foi treinado.
        """
        if not self.is_fitted:
            raise RuntimeError("Modelo não foi treinado")
        
        # Lidar com string individual
        if isinstance(X, str):
            X = [X]
            return_single = True
        else:
            return_single = False
        
        try:
            # Processar textos
            X_processed = [
                str(text).lower() if isinstance(text, str) else str(text).lower()
                for text in X
            ]
            
            # Vetorizar
            X_vectorized = self.vectorizer.transform(X_processed)
            
            # Obter probabilidades
            probas = self.model.predict_proba(X_vectorized)
            
            # Converter para dicionário
            results = []
            for proba in probas:
                proba_dict = {
                    cls: float(prob)
                    for cls, prob in zip(self.classes_, proba)
                }
                results.append(proba_dict)
            
            if return_single:
                return results[0]
            return results
            
        except Exception as e:
            logger.error(f"Erro durante predict_proba: {str(e)}")
            raise
    
    def evaluate(
        self,
        X_test: List[str],
        y_test: List[str],
        verbose: bool = True
    ) -> Dict[str, Union[float, np.ndarray, str]]:
        """
        Avalia o modelo em dados de teste.
        
        Args:
            X_test: Textos de teste.
            y_test: Labels de teste.
            verbose: Se True, imprime relatório completo.
        
        Returns:
            Dicionário com métricas de avaliação.
        """
        if not self.is_fitted:
            raise RuntimeError("Modelo não foi treinado")
        
        predictions = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision_macro': precision_score(
                y_test, predictions, average='macro', zero_division=0
            ),
            'recall_macro': recall_score(
                y_test, predictions, average='macro', zero_division=0
            ),
            'f1_macro': f1_score(
                y_test, predictions, average='macro', zero_division=0
            ),
            'precision_weighted': precision_score(
                y_test, predictions, average='weighted', zero_division=0
            ),
            'recall_weighted': recall_score(
                y_test, predictions, average='weighted', zero_division=0
            ),
            'f1_weighted': f1_score(
                y_test, predictions, average='weighted', zero_division=0
            ),
            'confusion_matrix': confusion_matrix(y_test, predictions),
            'classification_report': classification_report(
                y_test, predictions, zero_division=0
            ),
            'predictions': predictions
        }
        
        if verbose:
            logger.info("=" * 70)
            logger.info("RELATÓRIO DE AVALIAÇÃO")
            logger.info("=" * 70)
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Precision (macro): {metrics['precision_macro']:.4f}")
            logger.info(f"Recall (macro): {metrics['recall_macro']:.4f}")
            logger.info(f"F1-score (macro): {metrics['f1_macro']:.4f}")
            logger.info("\n" + metrics['classification_report'])
            logger.info("=" * 70)
        
        return metrics
    
    def save(self, model_path: str, vectorizer_path: str) -> None:
        """
        Salva o modelo e vectorizer em ficheiros.
        
        Args:
            model_path: Caminho para guardar o modelo.
            vectorizer_path: Caminho para guardar o vectorizer.
        
        Raises:
            RuntimeError: Se o modelo não foi treinado.
            IOError: Se houver erro ao guardar.
        """
        if not self.is_fitted:
            raise RuntimeError("Modelo não foi treinado. Use fit() primeiro.")
        
        try:
            # Criar directórios se não existirem
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)
            Path(vectorizer_path).parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.vectorizer, vectorizer_path)
            
            logger.info(f"Modelo guardado em: {model_path}")
            logger.info(f"Vectorizer guardado em: {vectorizer_path}")
            
        except Exception as e:
            logger.error(f"Erro ao guardar modelo: {str(e)}")
            raise IOError(f"Não foi possível guardar o modelo: {str(e)}")
    
    def load(self, model_path: str, vectorizer_path: str) -> None:
        """
        Carrega o modelo e vectorizer de ficheiros.
        
        Args:
            model_path: Caminho do modelo.
            vectorizer_path: Caminho do vectorizer.
        
        Raises:
            IOError: Se os ficheiros não forem encontrados.
        """
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.classes_ = self.model.classes_
            self.is_fitted = True
            
            logger.info(f"Modelo carregado de: {model_path}")
            logger.info(f"Vectorizer carregado de: {vectorizer_path}")
            
        except FileNotFoundError as e:
            logger.error(f"Ficheiro não encontrado: {str(e)}")
            raise IOError(f"Ficheiros do modelo não encontrados: {str(e)}")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise IOError(f"Não foi possível carregar o modelo: {str(e)}")
    
    def get_feature_importance(
        self,
        top_n: int = 10
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Retorna as features mais importantes por classe.
        
        Args:
            top_n: Número de top features a retornar por classe.
        
        Returns:
            Dicionário com top features por classe.
        
        Raises:
            RuntimeError: Se o modelo não foi treinado.
        """
        if not self.is_fitted:
            raise RuntimeError("Modelo não foi treinado")
        
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        feature_importance = {}
        
        for idx, class_label in enumerate(self.classes_):
            # Obter log-probabilidades
            log_probs = self.model.feature_log_prob_[idx]
            
            # Top features mais importantes
            top_indices = np.argsort(log_probs)[-top_n:][::-1]
            top_features = [
                (feature_names[i], float(log_probs[i]))
                for i in top_indices
            ]
            
            feature_importance[class_label] = top_features
        
        return feature_importance
    
    def __repr__(self) -> str:
        """Representação string do classificador."""
        status = "Treinado" if self.is_fitted else "Não treinado"
        return (
            f"NaiveBayesEmailClassifier("
            f"max_features={self.max_features}, "
            f"ngram_range={self.ngram_range}, "
            f"alpha={self.alpha}, "
            f"status={status})"
        )
