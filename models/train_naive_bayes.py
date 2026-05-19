# -*- coding: utf-8 -*-
"""
Script de Treino do Classificador Naive Bayes.

Este script carrega o dataset de emails, realiza preprocessing,
divide em treino/teste, treina o modelo Naive Bayes e realiza
avaliação experimental completa.

Uso:
    python train_naive_bayes.py
    python train_naive_bayes.py --dataset path/to/dataset.json
    python train_naive_bayes.py --max-features 10000 --alpha 0.5
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import sys

# Adicionar parent ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.model_selection import train_test_split, cross_val_score
from models.naive_bayes_classifier import NaiveBayesEmailClassifier
from models.utils import load_dataset, preprocess_texts, combine_text_fields


# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_classifier(
    dataset_path: str,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    alpha: float = 1.0,
    test_size: float = 0.2,
    random_state: int = 42,
    use_stopwords: bool = False,
    verbose: bool = True
) -> Tuple[NaiveBayesEmailClassifier, Dict]:
    """
    Treina o classificador Naive Bayes.
    
    Args:
        dataset_path: Caminho do dataset em JSON.
        max_features: Número máximo de features TF-IDF.
        ngram_range: Intervalo de n-gramas.
        alpha: Parâmetro de suavização Laplace.
        test_size: Proporção de dados para teste.
        random_state: Seed para reprodutibilidade.
        use_stopwords: Se usa stopwords em português.
        verbose: Se imprime informações detalhadas.
    
    Returns:
        Tupla (modelo_treinado, métricas_de_avaliação).
    """
    logger.info("=" * 80)
    logger.info("INICIANDO TREINO DO CLASSIFICADOR NAIVE BAYES")
    logger.info("=" * 80)
    
    # Carregar dataset
    logger.info(f"Carregando dataset de: {dataset_path}")
    emails = load_dataset(dataset_path)
    logger.info(f"Dataset carregado com {len(emails)} emails")
    
    # Extrair textos e labels
    logger.info("Extraindo textos e labels...")
    texts = []
    labels = []
    
    for email in emails:
        try:
            # Combinar subject e body
            combined_text = combine_text_fields(email)
            if combined_text:  # Apenas adicionar se não vazio
                texts.append(combined_text)
                label = email.get('label', 'desconhecido')
                labels.append(label)
        except Exception as e:
            logger.warning(f"Erro ao processar email: {str(e)}")
            continue
    
    logger.info(f"Total de emails processados: {len(texts)}")
    
    # Distribuição de classes
    unique_labels = list(set(labels))
    logger.info(f"Classes encontradas: {unique_labels}")
    for label in unique_labels:
        count = labels.count(label)
        percentage = (count / len(labels)) * 100
        logger.info(f"  - {label}: {count} ({percentage:.1f}%)")
    
    # Pré-processar textos
    logger.info("Realizando pré-processamento de textos...")
    texts = preprocess_texts(texts, remove_punctuation=False, lowercase=True)
    
    # Dividir em treino/teste
    logger.info(f"Dividindo em treino/teste com test_size={test_size}...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    
    logger.info(f"Treino: {len(X_train)} exemplos")
    logger.info(f"Teste: {len(X_test)} exemplos")
    
    # Criar e treinar modelo
    logger.info("\nCriando classificador...")
    clf = NaiveBayesEmailClassifier(
        max_features=max_features,
        ngram_range=ngram_range,
        alpha=alpha,
        use_stopwords=use_stopwords,
        random_state=random_state
    )
    
    logger.info("Treinando modelo...")
    clf.fit(X_train, y_train)
    
    # Avaliação
    logger.info("\nAvaliando modelo...")
    metrics = clf.evaluate(X_test, y_test, verbose=verbose)
    
    # Cross-validation
    logger.info("\nRealizando validação cruzada (5-fold)...")
    try:
        # Vetorizar textos para cross-validation
        X_vectorized = clf.vectorizer.transform(texts)
        cv_scores = cross_val_score(
            clf.model, X_vectorized, labels,
            cv=5, scoring='f1_weighted'
        )
        logger.info(f"Cross-validation F1-scores: {cv_scores}")
        logger.info(f"Média: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        metrics['cv_scores'] = cv_scores
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
    except Exception as e:
        logger.warning(f"Erro ao executar cross-validation: {str(e)}")
    
    # Feature importance
    logger.info("\nFeatures mais importantes por classe:")
    feature_importance = clf.get_feature_importance(top_n=10)
    for class_label, features in feature_importance.items():
        logger.info(f"\n  {class_label}:")
        for feature, score in features[:5]:
            logger.info(f"    - {feature}: {score:.4f}")
    
    metrics['feature_importance'] = feature_importance
    
    # Resumo
    logger.info("\n" + "=" * 80)
    logger.info("RESUMO DO TREINO")
    logger.info("=" * 80)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1-score (macro): {metrics['f1_macro']:.4f}")
    logger.info(f"F1-score (weighted): {metrics['f1_weighted']:.4f}")
    logger.info("=" * 80 + "\n")
    
    return clf, metrics


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Treina classificador Naive Bayes para emails em PT-PT"
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='dataset/dataset.json',
        help='Caminho do dataset em JSON'
    )
    parser.add_argument(
        '--max-features',
        type=int,
        default=5000,
        help='Número máximo de features TF-IDF'
    )
    parser.add_argument(
        '--ngrams',
        type=int,
        nargs=2,
        default=[1, 2],
        help='Intervalo de n-gramas (min max)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help='Parâmetro de suavização Laplace'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proporção de dados para teste'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Seed para reprodutibilidade'
    )
    parser.add_argument(
        '--stopwords',
        action='store_true',
        help='Usar stopwords em português'
    )
    parser.add_argument(
        '--output-model',
        type=str,
        default='models/naive_bayes_model.joblib',
        help='Caminho para guardar o modelo'
    )
    parser.add_argument(
        '--output-vectorizer',
        type=str,
        default='models/naive_bayes_vectorizer.joblib',
        help='Caminho para guardar o vectorizer'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Imprime detalhes de execução'
    )
    
    args = parser.parse_args()
    
    # Validar dataset
    if not Path(args.dataset).exists():
        logger.error(f"Dataset não encontrado: {args.dataset}")
        return
    
    # Treinar
    try:
        clf, metrics = train_classifier(
            dataset_path=args.dataset,
            max_features=args.max_features,
            ngram_range=tuple(args.ngrams),
            alpha=args.alpha,
            test_size=args.test_size,
            random_state=args.random_state,
            use_stopwords=args.stopwords,
            verbose=args.verbose
        )
        
        # Guardar modelo
        logger.info(f"Guardando modelo em: {args.output_model}")
        logger.info(f"Guardando vectorizer em: {args.output_vectorizer}")
        clf.save(args.output_model, args.output_vectorizer)
        
        logger.info("✓ Treino concluído com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro durante o treino: {str(e)}")
        raise


if __name__ == '__main__':
    main()
