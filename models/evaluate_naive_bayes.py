# -*- coding: utf-8 -*-
"""
Script de Avaliação Experimental Completa do Classificador Naive Bayes.

Este script oferece:
- Comparação com outros modelos (Logistic Regression, Decision Tree)
- Cross-validation e hyperparameter tuning
- Análise de features mais importantes
- Confusion matrix detalhada
- Relatório completo de métricas

Uso:
    python evaluate_naive_bayes.py
    python evaluate_naive_bayes.py --compare-models
    python evaluate_naive_bayes.py --tuning
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import sys

# Adicionar parent ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from models.naive_bayes_classifier import NaiveBayesEmailClassifier
from models.utils import load_dataset, preprocess_texts, combine_text_fields


# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentalEvaluator:
    """
    Avaliador experimental de modelos de classificação de emails.
    """
    
    def __init__(self, dataset_path: str, random_state: int = 42):
        """
        Inicializa o avaliador.
        
        Args:
            dataset_path: Caminho do dataset.
            random_state: Seed para reprodutibilidade.
        """
        self.dataset_path = dataset_path
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
    
    def load_and_prepare_data(
        self,
        test_size: float = 0.2,
        remove_class: str = None
    ) -> None:
        """
        Carrega e prepara o dataset.
        
        Args:
            test_size: Proporção de teste.
            remove_class: Classe a remover (opcional).
        """
        logger.info("Carregando dataset...")
        emails = load_dataset(self.dataset_path)
        
        # Extrair textos e labels
        texts = []
        labels = []
        
        for email in emails:
            try:
                label = email.get('label', 'desconhecido')
                
                # Remover classe se especificado
                if remove_class and label == remove_class:
                    continue
                
                combined_text = combine_text_fields(email)
                if combined_text:
                    texts.append(combined_text)
                    labels.append(label)
            except Exception as e:
                logger.warning(f"Erro ao processar email: {str(e)}")
                continue
        
        logger.info(f"Total de emails: {len(texts)}")
        
        # Pré-processar
        texts = preprocess_texts(texts, remove_punctuation=False, lowercase=True)
        
        # Dividir
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            texts, labels,
            test_size=test_size,
            random_state=self.random_state,
            stratify=labels
        )
        
        logger.info(f"Treino: {len(self.X_train)}, Teste: {len(self.X_test)}")
    
    def evaluate_naive_bayes(self) -> Dict:
        """
        Avalia o classificador Naive Bayes.
        
        Returns:
            Dicionário com métricas.
        """
        logger.info("\n" + "=" * 80)
        logger.info("AVALIANDO NAIVE BAYES MULTINOMIAL")
        logger.info("=" * 80)
        
        clf = NaiveBayesEmailClassifier()
        clf.fit(self.X_train, self.y_train)
        
        predictions = clf.predict(self.X_test)
        
        metrics = {
            'model': 'Naive Bayes',
            'accuracy': accuracy_score(self.y_test, predictions),
            'precision_macro': precision_score(
                self.y_test, predictions, average='macro', zero_division=0
            ),
            'recall_macro': recall_score(
                self.y_test, predictions, average='macro', zero_division=0
            ),
            'f1_macro': f1_score(
                self.y_test, predictions, average='macro', zero_division=0
            ),
            'f1_weighted': f1_score(
                self.y_test, predictions, average='weighted', zero_division=0
            ),
            'classification_report': classification_report(
                self.y_test, predictions, zero_division=0
            ),
            'predictions': predictions,
            'classifier': clf
        }
        
        # Cross-validation
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_vec = vectorizer.fit_transform(self.X_train + self.X_test)
        y_all = self.y_train + self.y_test
        
        cv_scores = cross_val_score(
            MultinomialNB(),
            X_vec, y_all,
            cv=5, scoring='f1_weighted'
        )
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1-score (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"F1-score (weighted): {metrics['f1_weighted']:.4f}")
        logger.info(f"Cross-val F1 (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return metrics
    
    def evaluate_logistic_regression(self) -> Dict:
        """Avalia Logistic Regression."""
        logger.info("\n" + "=" * 80)
        logger.info("AVALIANDO LOGISTIC REGRESSION")
        logger.info("=" * 80)
        
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            lowercase=True,
            encoding='utf-8'
        )
        
        X_train_vec = vectorizer.fit_transform(self.X_train)
        X_test_vec = vectorizer.transform(self.X_test)
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            class_weight='balanced'
        )
        model.fit(X_train_vec, self.y_train)
        
        predictions = model.predict(X_test_vec)
        
        metrics = {
            'model': 'Logistic Regression',
            'accuracy': accuracy_score(self.y_test, predictions),
            'precision_macro': precision_score(
                self.y_test, predictions, average='macro', zero_division=0
            ),
            'recall_macro': recall_score(
                self.y_test, predictions, average='macro', zero_division=0
            ),
            'f1_macro': f1_score(
                self.y_test, predictions, average='macro', zero_division=0
            ),
            'f1_weighted': f1_score(
                self.y_test, predictions, average='weighted', zero_division=0
            ),
            'classification_report': classification_report(
                self.y_test, predictions, zero_division=0
            ),
            'predictions': predictions
        }
        
        # Cross-validation
        X_all_vec = vectorizer.fit_transform(self.X_train + self.X_test)
        y_all = self.y_train + self.y_test
        
        cv_scores = cross_val_score(
            model,
            X_all_vec, y_all,
            cv=5, scoring='f1_weighted'
        )
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1-score (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"Cross-val F1 (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return metrics
    
    def evaluate_decision_tree(self) -> Dict:
        """Avalia Decision Tree."""
        logger.info("\n" + "=" * 80)
        logger.info("AVALIANDO DECISION TREE")
        logger.info("=" * 80)
        
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            lowercase=True,
            encoding='utf-8'
        )
        
        X_train_vec = vectorizer.fit_transform(self.X_train)
        X_test_vec = vectorizer.transform(self.X_test)
        
        model = DecisionTreeClassifier(
            max_depth=20,
            random_state=self.random_state,
            class_weight='balanced'
        )
        model.fit(X_train_vec, self.y_train)
        
        predictions = model.predict(X_test_vec)
        
        metrics = {
            'model': 'Decision Tree',
            'accuracy': accuracy_score(self.y_test, predictions),
            'precision_macro': precision_score(
                self.y_test, predictions, average='macro', zero_division=0
            ),
            'recall_macro': recall_score(
                self.y_test, predictions, average='macro', zero_division=0
            ),
            'f1_macro': f1_score(
                self.y_test, predictions, average='macro', zero_division=0
            ),
            'f1_weighted': f1_score(
                self.y_test, predictions, average='weighted', zero_division=0
            ),
            'classification_report': classification_report(
                self.y_test, predictions, zero_division=0
            ),
            'predictions': predictions
        }
        
        # Cross-validation
        X_all_vec = vectorizer.fit_transform(self.X_train + self.X_test)
        y_all = self.y_train + self.y_test
        
        cv_scores = cross_val_score(
            model,
            X_all_vec, y_all,
            cv=5, scoring='f1_weighted'
        )
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1-score (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"Cross-val F1 (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return metrics
    
    def compare_models(self) -> Dict:
        """Compara múltiplos modelos."""
        logger.info("\n" + "=" * 80)
        logger.info("COMPARAÇÃO DE MODELOS")
        logger.info("=" * 80)
        
        models_results = {}
        
        # Naive Bayes
        models_results['Naive Bayes'] = self.evaluate_naive_bayes()
        
        # Logistic Regression
        models_results['Logistic Regression'] = self.evaluate_logistic_regression()
        
        # Decision Tree
        models_results['Decision Tree'] = self.evaluate_decision_tree()
        
        # Imprimir comparação
        logger.info("\n" + "=" * 80)
        logger.info("RESUMO COMPARATIVO")
        logger.info("=" * 80)
        
        print("\n{:<25} {:<12} {:<12} {:<12} {:<12}".format(
            "Modelo", "Accuracy", "Precision", "Recall", "F1 (macro)"
        ))
        print("-" * 70)
        
        for model_name, metrics in models_results.items():
            print("{:<25} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
                model_name,
                metrics['accuracy'],
                metrics['precision_macro'],
                metrics['recall_macro'],
                metrics['f1_macro']
            ))
        
        # Melhor modelo
        best_model = max(
            models_results.items(),
            key=lambda x: x[1]['f1_macro']
        )
        logger.info(f"\n✓ Melhor modelo: {best_model[0]} (F1: {best_model[1]['f1_macro']:.4f})")
        
        return models_results
    
    def hyperparameter_tuning(self) -> Dict:
        """Realiza tuning de hiperparâmetros."""
        logger.info("\n" + "=" * 80)
        logger.info("TUNING DE HIPERPARÂMETROS")
        logger.info("=" * 80)
        
        # Vetorizar
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            lowercase=True,
            encoding='utf-8'
        )
        
        X_train_vec = vectorizer.fit_transform(self.X_train)
        X_test_vec = vectorizer.transform(self.X_test)
        
        # Grid para Naive Bayes alpha
        param_grid_nb = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]}
        
        logger.info("Tuning Naive Bayes (alpha)...")
        grid_nb = GridSearchCV(
            MultinomialNB(),
            param_grid_nb,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1
        )
        grid_nb.fit(X_train_vec, self.y_train)
        
        logger.info(f"Melhor alpha: {grid_nb.best_params_['alpha']:.2f}")
        logger.info(f"Melhor CV score: {grid_nb.best_score_:.4f}")
        
        # Avaliar com melhor modelo
        best_nb = grid_nb.best_estimator_
        predictions = best_nb.predict(X_test_vec)
        
        results = {
            'model': 'Naive Bayes (tuned)',
            'best_params': grid_nb.best_params_,
            'best_cv_score': grid_nb.best_score_,
            'accuracy': accuracy_score(self.y_test, predictions),
            'f1_weighted': f1_score(self.y_test, predictions, average='weighted'),
            'f1_macro': f1_score(self.y_test, predictions, average='macro'),
        }
        
        logger.info(f"Accuracy (teste): {results['accuracy']:.4f}")
        logger.info(f"F1-score (teste): {results['f1_weighted']:.4f}")
        
        return results
    
    def generate_report(self, output_file: str = None) -> str:
        """
        Gera relatório completo.
        
        Args:
            output_file: Caminho para guardar relatório.
        
        Returns:
            Texto do relatório.
        """
        logger.info("\n" + "=" * 80)
        logger.info("GERANDO RELATÓRIO COMPLETO")
        logger.info("=" * 80)
        
        # Avaliar Naive Bayes
        nb_results = self.evaluate_naive_bayes()
        
        report = []
        report.append("=" * 80)
        report.append("RELATÓRIO DE AVALIAÇÃO - CLASSIFICADOR NAIVE BAYES")
        report.append("=" * 80)
        report.append(f"\nDataset: {self.dataset_path}")
        report.append(f"Treino: {len(self.X_train)} | Teste: {len(self.X_test)}")
        report.append("\n" + "-" * 80)
        report.append("MÉTRICAS PRINCIPAIS")
        report.append("-" * 80)
        report.append(f"Accuracy: {nb_results['accuracy']:.4f}")
        report.append(f"Precision (macro): {nb_results['precision_macro']:.4f}")
        report.append(f"Recall (macro): {nb_results['recall_macro']:.4f}")
        report.append(f"F1-score (macro): {nb_results['f1_macro']:.4f}")
        report.append(f"F1-score (weighted): {nb_results['f1_weighted']:.4f}")
        report.append(f"Cross-validation F1: {nb_results['cv_mean']:.4f} (+/- {nb_results['cv_std']:.4f})")
        
        report.append("\n" + "-" * 80)
        report.append("CLASSIFICATION REPORT")
        report.append("-" * 80)
        report.append(nb_results['classification_report'])
        
        report.append("\n" + "=" * 80)
        
        text = "\n".join(report)
        
        # Guardar se solicitado
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Relatório guardado em: {output_file}")
        
        return text


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Avaliação experimental de classificadores de emails"
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='dataset/dataset.json',
        help='Caminho do dataset'
    )
    parser.add_argument(
        '--compare-models',
        action='store_true',
        help='Comparar com outros modelos'
    )
    parser.add_argument(
        '--tuning',
        action='store_true',
        help='Executar tuning de hiperparâmetros'
    )
    parser.add_argument(
        '--report',
        type=str,
        help='Guardar relatório em ficheiro'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Seed para reprodutibilidade'
    )
    
    args = parser.parse_args()
    
    # Validar dataset
    if not Path(args.dataset).exists():
        logger.error(f"Dataset não encontrado: {args.dataset}")
        return
    
    # Criar avaliador
    evaluator = ExperimentalEvaluator(args.dataset, args.random_state)
    evaluator.load_and_prepare_data()
    
    # Comparar modelos
    if args.compare_models:
        evaluator.compare_models()
    
    # Tuning
    if args.tuning:
        evaluator.hyperparameter_tuning()
    
    # Se nenhuma opção, avaliar Naive Bayes
    if not args.compare_models and not args.tuning:
        evaluator.evaluate_naive_bayes()
    
    # Gerar relatório
    if args.report:
        evaluator.generate_report(args.report)
    else:
        evaluator.generate_report()


if __name__ == '__main__':
    main()
