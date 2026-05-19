# -*- coding: utf-8 -*-
"""
Exemplos de Uso do Classificador Naive Bayes.

Este arquivo contém exemplos práticos de como usar o classificador
para diferentes tarefas: treino, predição, avaliação e comparação.

Exemplos incluem:
1. Treino básico
2. Predição simples
3. Predição batch
4. Avaliação completa
5. Feature importance
6. Save/Load
"""

import logging
import json
from pathlib import Path
import sys

# Adicionar parent ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.naive_bayes_classifier import NaiveBayesEmailClassifier
from models.utils import (
    load_dataset,
    preprocess_texts,
    combine_text_fields,
    get_class_distribution
)
from sklearn.model_selection import train_test_split

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def exemplo_1_treino_basico():
    """
    Exemplo 1: Treino básico do classificador.
    
    Demonstra como:
    - Carregar dataset
    - Criar classificador
    - Treinar
    - Fazer predições
    """
    print("\n" + "=" * 80)
    print("EXEMPLO 1: TREINO BÁSICO")
    print("=" * 80 + "\n")
    
    # Carregar dataset
    logger.info("Carregando dataset...")
    emails = load_dataset('dataset/dataset.json')
    
    # Extrair textos e labels
    texts = []
    labels = []
    for email in emails:
        text = combine_text_fields(email)
        if text:
            texts.append(text)
            labels.append(email.get('label', 'desconhecido'))
    
    logger.info(f"Total de emails: {len(texts)}")
    
    # Pré-processar
    texts = preprocess_texts(texts, lowercase=True)
    
    # Dividir em treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    
    logger.info(f"Treino: {len(X_train)} | Teste: {len(X_test)}")
    
    # Criar classificador
    clf = NaiveBayesEmailClassifier(max_features=5000, ngram_range=(1, 2))
    
    # Treinar
    logger.info("Treinando modelo...")
    clf.fit(X_train, y_train)
    logger.info("✓ Treino concluído")
    
    # Avaliar
    logger.info("Avaliando...")
    metrics = clf.evaluate(X_test, y_test, verbose=True)
    
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"F1-score (macro): {metrics['f1_macro']:.4f}")
    print(f"F1-score (weighted): {metrics['f1_weighted']:.4f}")


def exemplo_2_predicao_simples():
    """
    Exemplo 2: Predição em textos individuais.
    
    Demonstra como:
    - Carregar modelo treinado
    - Fazer predição individual
    - Obter confiança
    - Visualizar top-N predições
    """
    print("\n" + "=" * 80)
    print("EXEMPLO 2: PREDIÇÃO SIMPLES")
    print("=" * 80 + "\n")
    
    # Carregar modelo
    clf = NaiveBayesEmailClassifier()
    try:
        clf.load(
            'models/naive_bayes_model.joblib',
            'models/naive_bayes_vectorizer.joblib'
        )
        logger.info("✓ Modelo carregado")
    except Exception as e:
        logger.warning(f"Modelo não encontrado. Executar treino primeiro: {e}")
        return
    
    # Textos de teste
    test_emails = [
        "Olá, conseguimos reunir amanhã à tarde?",
        "A reunião de sexta fica cancelada",
        "Pode ser segunda à noite?",
        "Envio o relatório em anexo",
        "Confirmo a reunião para terça às 10h"
    ]
    
    for text in test_emails:
        print(f"\n📧 Email: {text[:60]}...")
        
        # Predição
        prediction = clf.predict(text)
        print(f"   Predição: {prediction}")
        
        # Confiança
        proba = clf.predict_proba(text)
        confidence = proba[prediction]
        print(f"   Confiança: {confidence:.2%}")
        
        # Top 3
        top_3 = sorted(proba.items(), key=lambda x: x[1], reverse=True)[:3]
        print("   Top 3:")
        for cls, prob in top_3:
            print(f"      - {cls}: {prob:.2%}")


def exemplo_3_predicao_batch():
    """
    Exemplo 3: Predição em múltiplos emails.
    
    Demonstra como:
    - Fazer batch predictions
    - Processar ficheiro JSON
    - Guardar resultados
    """
    print("\n" + "=" * 80)
    print("EXEMPLO 3: PREDIÇÃO BATCH")
    print("=" * 80 + "\n")
    
    # Carregar modelo
    clf = NaiveBayesEmailClassifier()
    try:
        clf.load(
            'models/naive_bayes_model.joblib',
            'models/naive_bayes_vectorizer.joblib'
        )
    except Exception as e:
        logger.warning(f"Modelo não encontrado: {e}")
        return
    
    # Carregar emails de teste
    test_emails = load_dataset('dataset/dataset.json')[:5]
    
    texts = [combine_text_fields(email) for email in test_emails]
    
    # Predições batch
    logger.info("Realizando predições batch...")
    predictions = clf.predict(texts)
    probabilities = clf.predict_proba(texts)
    
    # Apresentar resultados
    results = []
    for idx, (text, pred, proba) in enumerate(zip(texts, predictions, probabilities), 1):
        result = {
            'id': idx,
            'prediction': pred,
            'confidence': float(proba[pred]),
            'probabilities': {k: float(v) for k, v in proba.items()}
        }
        results.append(result)
        
        logger.info(f"Email {idx}: {pred} ({proba[pred]:.2%})")
    
    # Guardar resultados
    output_path = 'models/batch_predictions.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✓ Resultados guardados em: {output_path}")


def exemplo_4_feature_importance():
    """
    Exemplo 4: Análise de features importantes.
    
    Demonstra como:
    - Obter features mais importantes por classe
    - Visualizar palavras-chave
    - Entender o modelo
    """
    print("\n" + "=" * 80)
    print("EXEMPLO 4: FEATURE IMPORTANCE")
    print("=" * 80 + "\n")
    
    # Carregar modelo
    clf = NaiveBayesEmailClassifier()
    try:
        clf.load(
            'models/naive_bayes_model.joblib',
            'models/naive_bayes_vectorizer.joblib'
        )
    except Exception as e:
        logger.warning(f"Modelo não encontrado: {e}")
        return
    
    # Obter features importantes
    logger.info("Analisando features importantes...")
    feature_importance = clf.get_feature_importance(top_n=10)
    
    # Apresentar
    for class_label, features in feature_importance.items():
        print(f"\n{'='*60}")
        print(f"Classe: {class_label}")
        print(f"{'='*60}")
        
        for feature, score in features:
            print(f"  ✓ {feature:<30} (score: {score:>8.4f})")


def exemplo_5_save_load():
    """
    Exemplo 5: Guardar e carregar modelo.
    
    Demonstra como:
    - Treinar modelo
    - Guardar em ficheiros
    - Carregar e reutilizar
    """
    print("\n" + "=" * 80)
    print("EXEMPLO 5: SAVE/LOAD")
    print("=" * 80 + "\n")
    
    # Carregar dataset
    logger.info("Preparando dados...")
    emails = load_dataset('dataset/dataset.json')
    
    texts = []
    labels = []
    for email in emails:
        text = combine_text_fields(email)
        if text:
            texts.append(text)
            labels.append(email.get('label', 'desconhecido'))
    
    texts = preprocess_texts(texts, lowercase=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Treinar
    logger.info("Treinando modelo...")
    clf = NaiveBayesEmailClassifier()
    clf.fit(X_train, y_train)
    
    # Guardar
    model_path = 'models/naive_bayes_model_example.joblib'
    vectorizer_path = 'models/naive_bayes_vectorizer_example.joblib'
    
    logger.info(f"Guardando em: {model_path}")
    clf.save(model_path, vectorizer_path)
    
    # Carregar em novo objeto
    logger.info("Carregando modelo guardado...")
    clf_loaded = NaiveBayesEmailClassifier()
    clf_loaded.load(model_path, vectorizer_path)
    
    # Verificar que funciona
    predictions_original = clf.predict(X_test[:5])
    predictions_loaded = clf_loaded.predict(X_test[:5])
    
    matches = sum(p1 == p2 for p1, p2 in zip(predictions_original, predictions_loaded))
    logger.info(f"✓ Verificação: {matches}/5 predições correspondem")


def exemplo_6_distribuicao_classes():
    """
    Exemplo 6: Análise da distribuição de classes.
    
    Demonstra como:
    - Analisar balanceamento do dataset
    - Obter estatísticas por classe
    """
    print("\n" + "=" * 80)
    print("EXEMPLO 6: DISTRIBUIÇÃO DE CLASSES")
    print("=" * 80 + "\n")
    
    # Carregar dataset
    emails = load_dataset('dataset/dataset.json')
    
    labels = [email.get('label', 'desconhecido') for email in emails]
    
    # Distribuição
    logger.info("Distribuição de classes no dataset:")
    distribution = get_class_distribution(labels, verbose=True)
    
    # Estatísticas
    print(f"\nTotal de exemplos: {len(labels)}")
    print(f"Total de classes: {len(distribution)}")
    
    # Classe minoritária e maioritária
    if distribution:
        min_class = min(distribution.items(), key=lambda x: x[1]['count'])
        max_class = max(distribution.items(), key=lambda x: x[1]['count'])
        
        print(f"\nClasse minoritária: {min_class[0]} ({min_class[1]['count']} exemplos)")
        print(f"Classe maioritária: {max_class[0]} ({max_class[1]['count']} exemplos)")


def main():
    """Executa todos os exemplos."""
    print("\n" + "=" * 80)
    print("EXEMPLOS DE USO - CLASSIFICADOR NAIVE BAYES")
    print("=" * 80)
    
    try:
        # Exemplo 1: Treino
        exemplo_1_treino_basico()
        
        # Exemplo 2: Predição simples
        exemplo_2_predicao_simples()
        
        # Exemplo 3: Batch
        exemplo_3_predicao_batch()
        
        # Exemplo 4: Feature importance
        exemplo_4_feature_importance()
        
        # Exemplo 5: Save/Load
        exemplo_5_save_load()
        
        # Exemplo 6: Distribuição
        exemplo_6_distribuicao_classes()
        
        print("\n" + "=" * 80)
        print("✓ TODOS OS EXEMPLOS CONCLUÍDOS COM SUCESSO")
        print("=" * 80 + "\n")
        
    except Exception as e:
        logger.error(f"Erro na execução: {str(e)}")
        raise


if __name__ == '__main__':
    main()
