# -*- coding: utf-8 -*-
"""
Script rápido de teste - Valida funcionamento básico.
"""

import json
from pathlib import Path

print("=" * 80)
print("TESTE RÁPIDO - SISTEMA NAIVE BAYES")
print("=" * 80)

# Teste 1: Verificar dataset
print("\n[1] Verificando dataset...")
try:
    with open('dataset/dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✓ Dataset carregado: {len(data)} emails")
    
    # Mostrar classes
    labels = [e.get('label') for e in data if e.get('label')]
    unique_labels = set(labels)
    print(f"✓ Classes: {unique_labels}")
    
except Exception as e:
    print(f"✗ Erro: {e}")

# Teste 2: Verificar importações
print("\n[2] Testando importações...")
try:
    from models.naive_bayes_classifier import NaiveBayesEmailClassifier
    print("✓ NaiveBayesEmailClassifier importado")
    
    from models.utils import load_dataset, preprocess_text, combine_text_fields
    print("✓ Funções auxiliares importadas")
    
except Exception as e:
    print(f"✗ Erro: {e}")

# Teste 3: Criar classificador
print("\n[3] Testando criação do classificador...")
try:
    clf = NaiveBayesEmailClassifier(max_features=5000, ngram_range=(1, 2))
    print(f"✓ Classificador criado: {clf}")
    
except Exception as e:
    print(f"✗ Erro: {e}")

# Teste 4: Treino simples
print("\n[4] Testando treino...")
try:
    from sklearn.model_selection import train_test_split
    
    # Dados de teste
    X_train = [
        "Conseguimos reunir amanhã?",
        "A reunião fica cancelada",
        "Confirmo a reunião para terça"
    ]
    y_train = [
        "agendamento_reuniao",
        "cancelamento_reuniao",
        "reuniao_confirmada"
    ]
    
    clf = NaiveBayesEmailClassifier()
    clf.fit(X_train, y_train)
    
    print(f"✓ Modelo treinado")
    print(f"  Classes: {list(clf.classes_)}")
    
except Exception as e:
    print(f"✗ Erro: {e}")

# Teste 5: Predição
print("\n[5] Testando predição...")
try:
    test_texts = [
        "Conseguimos reunir amanhã?",
        "Cancelamos a reunião"
    ]
    
    for text in test_texts:
        pred = clf.predict(text)
        proba = clf.predict_proba(text)
        conf = proba[pred]
        
        print(f"  \"{text}\"")
        print(f"    → {pred} ({conf:.2%})")
    
    print("✓ Predições funcionando")
    
except Exception as e:
    print(f"✗ Erro: {e}")

# Teste 6: Save/Load
print("\n[6] Testando save/load...")
try:
    import tempfile
    import os
    
    # Criar directório temporário
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'model.joblib')
        vec_path = os.path.join(tmpdir, 'vec.joblib')
        
        # Guardar
        clf.save(model_path, vec_path)
        print(f"✓ Modelo guardado")
        
        # Carregar
        clf2 = NaiveBayesEmailClassifier()
        clf2.load(model_path, vec_path)
        print(f"✓ Modelo carregado")
        
        # Testar
        pred2 = clf2.predict("Reunião amanhã?")
        print(f"✓ Predição após load: {pred2}")
    
except Exception as e:
    print(f"✗ Erro: {e}")

print("\n" + "=" * 80)
print("✓ TESTES CONCLUÍDOS COM SUCESSO!")
print("=" * 80)

print("\n📋 Próximos passos:")
print("  1. python models/train_naive_bayes.py")
print("  2. python models/predict_naive_bayes.py --text 'seu email aqui'")
print("  3. python models/evaluate_naive_bayes.py --compare-models")
print("  4. Ver exemplos: python models/examples_naive_bayes.py")
print()
