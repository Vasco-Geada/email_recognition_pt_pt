# -*- coding: utf-8 -*-
"""
Script de Teste - Validação do Sistema Naive Bayes.

Testa:
- Carregamento de dataset
- Pré-processamento
- Treino básico
- Predições
- Save/Load
- Feature importance
"""

import sys
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Adicionar parent ao path para importações
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Testa se todos os imports funcionam."""
    logger.info("Testando imports...")
    
    try:
        from models.naive_bayes_classifier import NaiveBayesEmailClassifier
        from models.utils import (
            load_dataset,
            preprocess_text,
            combine_text_fields
        )
        logger.info("✓ Imports OK")
        return True
    except Exception as e:
        logger.error(f"✗ Erro ao importar: {str(e)}")
        return False


def test_dataset_loading():
    """Testa carregamento de dataset."""
    logger.info("\nTestando carregamento de dataset...")
    
    try:
        from models.utils import load_dataset
        
        emails = load_dataset('dataset/dataset.json')
        
        if not emails:
            logger.error("✗ Dataset vazio")
            return False
        
        logger.info(f"✓ Dataset carregado: {len(emails)} emails")
        return True
        
    except Exception as e:
        logger.error(f"✗ Erro ao carregar dataset: {str(e)}")
        return False


def test_preprocessing():
    """Testa pré-processamento."""
    logger.info("\nTestando pré-processamento...")
    
    try:
        from models.utils import preprocess_text, combine_text_fields
        
        test_email = {
            'subject': 'Reunião amanhã',
            'body': 'Olá, consegues reunir amanhã à tarde?'
        }
        
        # Test combine
        combined = combine_text_fields(test_email)
        if not combined:
            logger.error("✗ Combinação de campos falhou")
            return False
        
        # Test preprocess
        cleaned = preprocess_text(combined)
        if not cleaned:
            logger.error("✗ Pré-processamento falhou")
            return False
        
        logger.info(f"✓ Pré-processamento OK")
        logger.info(f"  Original: {combined[:50]}...")
        logger.info(f"  Processado: {cleaned[:50]}...")
        return True
        
    except Exception as e:
        logger.error(f"✗ Erro no pré-processamento: {str(e)}")
        return False


def test_classifier_creation():
    """Testa criação do classificador."""
    logger.info("\nTestando criação do classificador...")
    
    try:
        from models.naive_bayes_classifier import NaiveBayesEmailClassifier
        
        clf = NaiveBayesEmailClassifier(
            max_features=5000,
            ngram_range=(1, 2),
            alpha=1.0
        )
        
        if not clf:
            logger.error("✗ Criação do classificador falhou")
            return False
        
        logger.info("✓ Classificador criado OK")
        logger.info(f"  {clf}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Erro ao criar classificador: {str(e)}")
        return False


def test_training():
    """Testa treino do modelo."""
    logger.info("\nTestando treino...")
    
    try:
        from models.naive_bayes_classifier import NaiveBayesEmailClassifier
        from models.utils import load_dataset, preprocess_texts, combine_text_fields
        
        # Carregar dados
        emails = load_dataset('dataset/dataset.json')
        texts = [combine_text_fields(e) for e in emails if combine_text_fields(e)]
        labels = [e.get('label', 'desconhecido') for e in emails][:len(texts)]
        
        if len(texts) < 10:
            logger.warning(f"⚠ Dataset pequeno: {len(texts)} exemplos")
        
        # Pré-processar
        texts = preprocess_texts(texts)
        
        # Treinar
        clf = NaiveBayesEmailClassifier()
        clf.fit(texts, labels)
        
        if not clf.is_fitted:
            logger.error("✗ Modelo não foi treinado")
            return False
        
        logger.info(f"✓ Treino OK")
        logger.info(f"  Classes: {list(clf.classes_)}")
        logger.info(f"  Exemplos: {len(texts)}")
        return True, clf, texts, labels
        
    except Exception as e:
        logger.error(f"✗ Erro ao treinar: {str(e)}")
        return False


def test_prediction(clf, texts):
    """Testa predição."""
    logger.info("\nTestando predição...")
    
    try:
        # Predição simples
        if len(texts) > 0:
            pred = clf.predict(texts[0])
            
            if not pred:
                logger.error("✗ Predição retornou vazio")
                return False
            
            logger.info(f"✓ Predição OK")
            logger.info(f"  Texto: {texts[0][:50]}...")
            logger.info(f"  Predição: {pred}")
            
            # Predição batch
            if len(texts) > 1:
                preds = clf.predict(texts[:3])
                logger.info(f"✓ Batch OK: {len(preds)} predições")
            
            return True
        else:
            logger.warning("⚠ Sem textos para testar predição")
            return True
        
    except Exception as e:
        logger.error(f"✗ Erro na predição: {str(e)}")
        return False


def test_probabilities(clf, texts):
    """Testa probabilidades."""
    logger.info("\nTestando probabilidades...")
    
    try:
        if len(texts) > 0:
            probs = clf.predict_proba(texts[0])
            
            if not probs:
                logger.error("✗ Probabilidades vazias")
                return False
            
            total_prob = sum(probs.values())
            if abs(total_prob - 1.0) > 0.01:
                logger.error(f"✗ Probabilidades não somam 1.0: {total_prob}")
                return False
            
            logger.info(f"✓ Probabilidades OK")
            for cls, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {cls}: {prob:.4f}")
            
            return True
        else:
            logger.warning("⚠ Sem textos para testar probabilidades")
            return True
        
    except Exception as e:
        logger.error(f"✗ Erro ao calcular probabilidades: {str(e)}")
        return False


def test_save_load(clf):
    """Testa save/load."""
    logger.info("\nTestando save/load...")
    
    try:
        from models.naive_bayes_classifier import NaiveBayesEmailClassifier
        
        model_path = 'models/test_model.joblib'
        vectorizer_path = 'models/test_vectorizer.joblib'
        
        # Guardar
        clf.save(model_path, vectorizer_path)
        
        # Verificar ficheiros
        if not Path(model_path).exists():
            logger.error(f"✗ Ficheiro de modelo não criado")
            return False
        
        if not Path(vectorizer_path).exists():
            logger.error(f"✗ Ficheiro de vectorizer não criado")
            return False
        
        # Carregar
        clf_loaded = NaiveBayesEmailClassifier()
        clf_loaded.load(model_path, vectorizer_path)
        
        if not clf_loaded.is_fitted:
            logger.error("✗ Modelo carregado não está treinado")
            return False
        
        logger.info(f"✓ Save/Load OK")
        logger.info(f"  Modelo: {model_path}")
        logger.info(f"  Vectorizer: {vectorizer_path}")
        
        # Limpar
        Path(model_path).unlink()
        Path(vectorizer_path).unlink()
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Erro no save/load: {str(e)}")
        return False


def test_feature_importance(clf):
    """Testa feature importance."""
    logger.info("\nTestando feature importance...")
    
    try:
        features = clf.get_feature_importance(top_n=5)
        
        if not features:
            logger.error("✗ Sem features")
            return False
        
        logger.info(f"✓ Feature importance OK")
        for cls, top_features in features.items():
            logger.info(f"  {cls}:")
            for feature, score in top_features[:3]:
                logger.info(f"    - {feature}: {score:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Erro ao obter feature importance: {str(e)}")
        return False


def main():
    """Executa todos os testes."""
    logger.info("=" * 80)
    logger.info("TESTES - SISTEMA NAIVE BAYES")
    logger.info("=" * 80)
    
    results = {}
    
    # Teste 1: Imports
    results['imports'] = test_imports()
    
    # Teste 2: Dataset
    results['dataset'] = test_dataset_loading()
    
    # Teste 3: Preprocessing
    results['preprocessing'] = test_preprocessing()
    
    # Teste 4: Classifier creation
    results['classifier'] = test_classifier_creation()
    
    # Teste 5: Training
    train_result = test_training()
    if isinstance(train_result, tuple):
        results['training'] = train_result[0]
        clf = train_result[1]
        texts = train_result[2]
        labels = train_result[3]
    else:
        results['training'] = train_result
        clf = None
    
    # Testes seguintes dependem do treino
    if results.get('training') and clf:
        results['prediction'] = test_prediction(clf, texts)
        results['probabilities'] = test_probabilities(clf, texts)
        results['save_load'] = test_save_load(clf)
        results['features'] = test_feature_importance(clf)
    else:
        logger.warning("⚠ Saltando testes dependentes (treino falhou)")
        results['prediction'] = False
        results['probabilities'] = False
        results['save_load'] = False
        results['features'] = False
    
    # Resumo
    logger.info("\n" + "=" * 80)
    logger.info("RESUMO DOS TESTES")
    logger.info("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status:8} - {test_name}")
    
    logger.info("=" * 80)
    logger.info(f"Resultado: {passed}/{total} testes passou")
    
    if passed == total:
        logger.info("✓ TODOS OS TESTES PASSARAM!")
        return 0
    else:
        logger.error(f"✗ {total - passed} teste(s) falharam")
        return 1


if __name__ == '__main__':
    sys.exit(main())
