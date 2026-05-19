# -*- coding: utf-8 -*-
"""
Módulo de Classificação de Emails com Naive Bayes.

Este pacote fornece uma implementação completa de classificação de emails
usando Naive Bayes Multinomial com TF-IDF para português europeu.

Módulos:
    naive_bayes_classifier: Classificador principal
    train_naive_bayes: Script de treino
    predict_naive_bayes: Script de predição
    evaluate_naive_bayes: Script de avaliação
    utils: Funções auxiliares

Exemplo:
    from models.naive_bayes_classifier import NaiveBayesEmailClassifier
    
    clf = NaiveBayesEmailClassifier()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
"""

__version__ = "1.0.0"
__author__ = "NLP Research Team"
__all__ = [
    'NaiveBayesEmailClassifier',
    'load_dataset',
    'preprocess_text',
    'combine_text_fields'
]

from .naive_bayes_classifier import NaiveBayesEmailClassifier
from .utils import (
    load_dataset,
    preprocess_text,
    preprocess_texts,
    combine_text_fields,
    clean_text,
    remove_email_signatures,
    remove_threads
)
