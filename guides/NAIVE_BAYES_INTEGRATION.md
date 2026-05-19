# Guia de Integração - Módulo Naive Bayes

## 📌 Visão Geral

Este guia explica como integrar o módulo Naive Bayes com o pipeline existente de processamento de emails.

## 🔗 Integração com Pipeline Existente

### 1. Usando com `preprocessing/preprocess.py`

O sistema Naive Bayes é compatível com o módulo de pré-processamento existente:

```python
import preprocessing.preprocess as prep
from models.naive_bayes_classifier import NaiveBayesEmailClassifier
from models.utils import combine_text_fields

# Usar pipeline existente
email = {
    "subject": "Reunião amanhã",
    "body": "Olá, consegues reunir?",
    "label": "agendamento_reuniao"
}

# Pré-processar com função existente
processed_email = prep.preprocessEmail(email)

# Extrair texto limpo
text = processed_email['clean_body']

# Usar classificador
clf = NaiveBayesEmailClassifier()
clf.load('models/naive_bayes_model.joblib', 
         'models/naive_bayes_vectorizer.joblib')

prediction = clf.predict(text)
confidence = clf.predict_proba(text)

print(f"Classe: {prediction}")
print(f"Confiança: {confidence[prediction]:.2%}")
```

### 2. Adicionar ao Pipeline de Análise

Criar um módulo que integra Naive Bayes no pipeline:

```python
# preprocessing/intent_classifier.py
import preprocessing.preprocess as prep
from models.naive_bayes_classifier import NaiveBayesEmailClassifier

class IntentClassifier:
    """Classificador de intenção integrado no pipeline."""
    
    def __init__(self, model_path, vectorizer_path):
        self.classifier = NaiveBayesEmailClassifier()
        self.classifier.load(model_path, vectorizer_path)
    
    def classify_email(self, email):
        """Classifica email e adiciona intenção aos metadados."""
        # Pré-processar com pipeline existente
        processed = prep.preprocessEmail(email)
        
        # Classificar
        text = processed['clean_body']
        intent = self.classifier.predict(text)
        confidence = self.classifier.predict_proba(text)
        
        # Adicionar aos metadados
        processed['intent'] = intent
        processed['intent_confidence'] = float(confidence[intent])
        processed['intent_probabilities'] = {
            k: float(v) for k, v in confidence.items()
        }
        
        return processed

# Uso
classifier = IntentClassifier(
    'models/naive_bayes_model.joblib',
    'models/naive_bayes_vectorizer.joblib'
)

email = {"subject": "Reunião", "body": "Conseguimos reunir?"}
result = classifier.classify_email(email)
print(result['intent'])  # 'agendamento_reuniao'
```

### 3. Integração com QA Pipeline

Usar Naive Bayes como classificador de intenção no módulo QA:

```python
# qa/qa_pipeline_enhanced.py
from models.naive_bayes_classifier import NaiveBayesEmailClassifier
from models.utils import combine_text_fields

class QAPipelineWithIntentClassification:
    """Pipeline QA com classificação de intenção Naive Bayes."""
    
    def __init__(self, model_path, vectorizer_path):
        self.intent_classifier = NaiveBayesEmailClassifier()
        self.intent_classifier.load(model_path, vectorizer_path)
    
    def process_email(self, email):
        """Processa email com classificação de intenção."""
        # Classificar intenção
        text = combine_text_fields(email)
        intent = self.intent_classifier.predict(text)
        confidence = self.intent_classifier.predict_proba(text)
        
        # Gerar QA baseado em intenção
        qa_pairs = self._generate_qa_by_intent(
            email, intent, confidence
        )
        
        return {
            'intent': intent,
            'confidence': float(confidence[intent]),
            'qa_pairs': qa_pairs
        }
    
    def _generate_qa_by_intent(self, email, intent, confidence):
        """Gera QA específicas por tipo de intenção."""
        questions = {
            'agendamento_reuniao': [
                'Quando o utilizador propõe reunir?',
                'Qual é a hora sugerida?',
                'Qual é o local da reunião?'
            ],
            'cancelamento_reuniao': [
                'Qual é a data da reunião cancelada?',
                'Qual é a razão do cancelamento?',
                'Será agendada nova reunião?'
            ],
            'reuniao_confirmada': [
                'Qual é a data confirmada?',
                'A que horas é a reunião?',
                'Onde será realizada?'
            ]
        }
        
        return questions.get(intent, [])
```

### 4. Comparação de Modelos em Pipeline

Usar múltiplos modelos para decisão robusta:

```python
# models/ensemble_classifier.py
from models.naive_bayes_classifier import NaiveBayesEmailClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from models.utils import combine_text_fields
import numpy as np

class EnsembleEmailClassifier:
    """Ensemble de múltiplos modelos."""
    
    def __init__(self, nb_model_path, nb_vectorizer_path):
        self.nb_classifier = NaiveBayesEmailClassifier()
        self.nb_classifier.load(nb_model_path, nb_vectorizer_path)
    
    def predict_with_confidence(self, text, method='ensemble'):
        """Predição com múltiplos métodos."""
        
        if method == 'naive_bayes':
            return self._predict_nb(text)
        elif method == 'ensemble':
            return self._predict_ensemble(text)
    
    def _predict_nb(self, text):
        pred = self.nb_classifier.predict(text)
        proba = self.nb_classifier.predict_proba(text)
        return {
            'model': 'Naive Bayes',
            'prediction': pred,
            'confidence': float(proba[pred]),
            'probabilities': {k: float(v) for k, v in proba.items()}
        }
    
    def _predict_ensemble(self, text):
        """Combina predições de múltiplos modelos."""
        # Obter predição NB
        nb_result = self._predict_nb(text)
        
        # Aqui poderiam ser adicionados mais modelos
        # Combinar resultados (votação, média, etc.)
        
        return {
            'model': 'Ensemble',
            'prediction': nb_result['prediction'],
            'confidence': nb_result['confidence'],
            'methods': [nb_result]
        }
```

## 🔄 Fluxo de Integração Completo

```
Email Original
    ↓
├─ preprocessing/preprocess.py (limpeza)
│   ├─ normalizeText
│   ├─ removeEmailHistory
│   └─ removeSignature
│
├─ models/naive_bayes_classifier.py (classificação)
│   ├─ TF-IDF vectorização
│   ├─ Predição
│   └─ Confiança
│
├─ qa/qa_pipeline.py (QA baseado em intenção)
│   └─ Gerar perguntas/respostas
│
└─ Resultado Final
    ├─ Intent
    ├─ Confidence
    ├─ QA Pairs
    └─ Features Importantes
```

## 📊 Exemplo Completo de Integração

```python
# main_pipeline.py
import preprocessing.preprocess as prep
from models.naive_bayes_classifier import NaiveBayesEmailClassifier
from models.utils import load_dataset, combine_text_fields

def full_email_processing_pipeline(email_dict):
    """Pipeline completo de processamento de emails."""
    
    # 1. Pré-processamento
    print("1. Pré-processando email...")
    processed = prep.preprocessEmail(email_dict)
    print(f"   ✓ Texto limpo: {processed['clean_body'][:50]}...")
    
    # 2. Classificação de intenção
    print("\n2. Classificando intenção...")
    classifier = NaiveBayesEmailClassifier()
    classifier.load(
        'models/naive_bayes_model.joblib',
        'models/naive_bayes_vectorizer.joblib'
    )
    
    text = combine_text_fields(email_dict)
    intent = classifier.predict(text)
    confidence = classifier.predict_proba(text)
    
    print(f"   ✓ Intenção: {intent}")
    print(f"   ✓ Confiança: {confidence[intent]:.2%}")
    
    # 3. Análise de features
    print("\n3. Features importantes...")
    features = classifier.get_feature_importance(top_n=5)
    for cls, top_features in features.items():
        if cls == intent:
            print(f"   Para '{intent}':")
            for feature, score in top_features[:3]:
                print(f"      - {feature}: {score:.4f}")
    
    # 4. Resultado final
    result = {
        'original': email_dict,
        'processed': processed,
        'classification': {
            'intent': intent,
            'confidence': float(confidence[intent]),
            'all_probabilities': {k: float(v) for k, v in confidence.items()}
        }
    }
    
    return result

# Uso
if __name__ == '__main__':
    email = {
        'subject': 'Reunião amanhã',
        'body': 'Olá, consegues reunir amanhã à tarde?'
    }
    
    result = full_email_processing_pipeline(email)
    print("\n" + "=" * 70)
    print("RESULTADO FINAL")
    print("=" * 70)
    print(f"Intenção: {result['classification']['intent']}")
    print(f"Confiança: {result['classification']['confidence']:.2%}")
```

## 🚀 Performance e Otimizações

### Cache de Modelos

Para aplicações com múltiplas predições, cache o modelo carregado:

```python
class CachedClassifier:
    """Classificador com cache de modelo."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CachedClassifier, cls).__new__(cls)
            cls._instance.classifier = NaiveBayesEmailClassifier()
            cls._instance.classifier.load(
                'models/naive_bayes_model.joblib',
                'models/naive_bayes_vectorizer.joblib'
            )
        return cls._instance
    
    def predict(self, text):
        return self.classifier.predict(text)

# Uso
clf = CachedClassifier()
pred1 = clf.predict(email1)  # Carrega modelo
pred2 = clf.predict(email2)  # Usa cache
```

### Batch Processing

Para processar múltiplos emails:

```python
def process_emails_batch(emails_list, batch_size=100):
    """Processa emails em batches."""
    
    classifier = CachedClassifier()
    results = []
    
    for i in range(0, len(emails_list), batch_size):
        batch = emails_list[i:i+batch_size]
        texts = [combine_text_fields(e) for e in batch]
        
        predictions = classifier.predict(texts)
        
        for email, pred in zip(batch, predictions):
            results.append({
                'email_id': email.get('email_id'),
                'intent': pred
            })
    
    return results
```

## ⚙️ Configuração Recomendada

```python
# config.py
NAIVE_BAYES_CONFIG = {
    'model_path': 'models/naive_bayes_model.joblib',
    'vectorizer_path': 'models/naive_bayes_vectorizer.joblib',
    'max_features': 5000,
    'ngram_range': (1, 2),
    'alpha': 1.0,
    'confidence_threshold': 0.7,  # Threshold mínimo
    'use_cache': True  # Cache de modelo
}

# Uso
from config import NAIVE_BAYES_CONFIG

classifier = NaiveBayesEmailClassifier(
    max_features=NAIVE_BAYES_CONFIG['max_features'],
    ngram_range=NAIVE_BAYES_CONFIG['ngram_range'],
    alpha=NAIVE_BAYES_CONFIG['alpha']
)

classifier.load(
    NAIVE_BAYES_CONFIG['model_path'],
    NAIVE_BAYES_CONFIG['vectorizer_path']
)

prediction = classifier.predict(text)
proba = classifier.predict_proba(text)

# Verificar threshold
if proba[prediction] >= NAIVE_BAYES_CONFIG['confidence_threshold']:
    use_prediction(prediction)
else:
    flag_for_manual_review(text, proba)
```

## 📝 Checklist de Integração

- [ ] Verificar que dataset está em `dataset/dataset.json`
- [ ] Treinar modelo: `python models/train_naive_bayes.py`
- [ ] Testar sistema: `python models/test_naive_bayes.py`
- [ ] Carregar módulo em novo script
- [ ] Testar predições em exemplos PT-PT
- [ ] Avaliar performance comparado com baseline
- [ ] Integrar com pipeline QA
- [ ] Documentar mudanças

## 🔍 Troubleshooting de Integração

### Problema: Incompatibilidade de Encoding

**Solução:**
```python
# Garantir UTF-8
text = email_text.encode('utf-8').decode('utf-8')
prediction = classifier.predict(text)
```

### Problema: Modelo não carrega

**Solução:**
```python
from pathlib import Path

# Verificar caminhos
model_path = Path('models/naive_bayes_model.joblib').resolve()
if not model_path.exists():
    print(f"Treinar modelo: python models/train_naive_bayes.py")
else:
    classifier.load(str(model_path), ...)
```

### Problema: Performance lenta

**Solução:**
```python
# Usar cache singleton
classifier = CachedClassifier()

# Reduzir max_features
classifier = NaiveBayesEmailClassifier(max_features=2000)

# Usar batch processing
predictions = classifier.predict(texts_list)  # Mais rápido que um por um
```

## 📚 Próximos Passos

1. **Treinar modelo inicial**
   ```bash
   python models/train_naive_bayes.py
   ```

2. **Testar sistema**
   ```bash
   python models/test_naive_bayes.py
   ```

3. **Avaliar performance**
   ```bash
   python models/evaluate_naive_bayes.py --compare-models
   ```

4. **Integrar com QA pipeline**
   - Adicionar classificação de intenção às respostas
   - Usar intenção para refinar QA pairs

5. **Otimizar e tunar**
   ```bash
   python models/evaluate_naive_bayes.py --tuning
   ```

## 📞 Suporte

Para questões de integração, consultar:
- `models/README_NAIVE_BAYES.md` - Documentação completa
- `models/examples_naive_bayes.py` - Exemplos práticos
- `guides/` - Documentação geral do projeto
