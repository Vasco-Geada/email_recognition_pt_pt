# 📧 Classificador de Emails com Naive Bayes (PT-PT)

## Visão Geral

Sistema completo de classificação de emails em português europeu utilizando **Naive Bayes Multinomial** com **TF-IDF**. Designed para reconhecer intenções em emails académicos informais relacionados com reuniões.

### Classes de Classificação

- `agendamento_reuniao` - Emails solicitando agendamento de reunião
- `cancelamento_reuniao` - Emails cancelando reuniões
- `reuniao_confirmada` - Emails confirmando reunião
- `nao_reuniao` - Emails sem relação com reuniões

## 🏗️ Arquitetura

```
models/
├── naive_bayes_classifier.py      # Classificador principal
├── train_naive_bayes.py           # Script de treino
├── predict_naive_bayes.py         # Script de predição
├── evaluate_naive_bayes.py        # Script de avaliação
├── examples_naive_bayes.py        # Exemplos de uso
├── utils.py                       # Funções auxiliares
├── __init__.py                    # Importações
└── README.md                      # Este ficheiro
```

## 📋 Requisitos

```
scikit-learn>=1.0.0
numpy>=1.20.0
joblib>=1.0.0
```

## 🚀 Quick Start

### 1. Treino Básico

```bash
python models/train_naive_bayes.py --dataset dataset/dataset.json
```

**Output esperado:**
- Modelo treinado em `models/naive_bayes_model.joblib`
- Vectorizer em `models/naive_bayes_vectorizer.joblib`
- Relatório de métricas

### 2. Predição Simples

```bash
python models/predict_naive_bayes.py --text "Olá, conseguimos reunir amanhã?"
```

**Output esperado:**
```
======================================================================
📧 Predição: agendamento_reuniao
📊 Confiança: 91.23%
======================================================================
```

### 3. Predição Batch

```bash
python models/predict_naive_bayes.py --file dataset/dataset.json --batch --output predictions.json
```

### 4. Avaliação Experimental

```bash
python models/evaluate_naive_bayes.py --compare-models --tuning
```

## 📚 API Detalhada

### NaiveBayesEmailClassifier

Classe principal para classificação de emails.

#### Inicialização

```python
from models.naive_bayes_classifier import NaiveBayesEmailClassifier

clf = NaiveBayesEmailClassifier(
    max_features=5000,           # Máximo de features TF-IDF
    ngram_range=(1, 2),          # Uni-gramas e bi-gramas
    alpha=1.0,                   # Suavização Laplace
    use_stopwords=False,         # Usar stopwords PT
    min_df=1,                    # Frequência mínima
    max_df=1.0,                  # Proporção máxima
    random_state=42              # Reprodutibilidade
)
```

#### Treino

```python
from models.utils import load_dataset, preprocess_texts, combine_text_fields
from sklearn.model_selection import train_test_split

# Carregar e preparar dados
emails = load_dataset('dataset/dataset.json')
texts = [combine_text_fields(e) for e in emails]
labels = [e['label'] for e in emails]
texts = preprocess_texts(texts)

# Dividir
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

# Treinar
clf = NaiveBayesEmailClassifier()
clf.fit(X_train, y_train)
```

#### Predição

```python
# Predição simples
prediction = clf.predict("Conseguimos reunir amanhã?")
# Output: 'agendamento_reuniao'

# Predição batch
predictions = clf.predict([email1, email2, email3])
# Output: ['agendamento_reuniao', 'cancelamento_reuniao', 'nao_reuniao']

# Com confiança
probabilities = clf.predict_proba("Conseguimos reunir amanhã?")
# Output: {'agendamento_reuniao': 0.91, 'cancelamento_reuniao': 0.05, ...}
```

#### Avaliação

```python
metrics = clf.evaluate(X_test, y_test, verbose=True)

# Métricas retornadas:
# - accuracy
# - precision_macro, precision_weighted
# - recall_macro, recall_weighted
# - f1_macro, f1_weighted
# - confusion_matrix
# - classification_report
```

#### Feature Importance

```python
# Top 10 features por classe
features = clf.get_feature_importance(top_n=10)

for class_label, top_features in features.items():
    print(f"\nClasse: {class_label}")
    for feature, score in top_features:
        print(f"  {feature}: {score:.4f}")
```

#### Persistência

```python
# Guardar
clf.save(
    'models/naive_bayes_model.joblib',
    'models/naive_bayes_vectorizer.joblib'
)

# Carregar
clf = NaiveBayesEmailClassifier()
clf.load(
    'models/naive_bayes_model.joblib',
    'models/naive_bayes_vectorizer.joblib'
)
```

### Funções Auxiliares (utils.py)

#### load_dataset

```python
from models.utils import load_dataset

emails = load_dataset('dataset/dataset.json')
# Output: Lista de dicionários com emails
```

#### preprocess_text / preprocess_texts

```python
from models.utils import preprocess_text, preprocess_texts

# Texto único
clean = preprocess_text(
    "Olá, conseguimos reunir amanhã?",
    remove_signatures=True,
    remove_threads_history=True,
    lowercase=True
)

# Múltiplos textos
texts = ["Email 1", "Email 2", "Email 3"]
clean_texts = preprocess_texts(texts)
```

#### combine_text_fields

```python
from models.utils import combine_text_fields

email_dict = {
    'subject': 'Reunião amanhã',
    'body': 'Consegues vir à tarde?',
    'label': 'agendamento_reuniao'
}

combined = combine_text_fields(
    email_dict,
    subject_weight=1.0,
    body_weight=2.0
)
# Output: "Reunião amanhã Consegues vir à tarde? Consegues vir à tarde?"
```

#### get_class_distribution

```python
from models.utils import get_class_distribution

labels = ['agendamento_reuniao', 'agendamento_reuniao', 'cancelamento_reuniao']
distribution = get_class_distribution(labels)

# Output:
# {
#   'agendamento_reuniao': {'count': 2, 'percentage': 66.67},
#   'cancelamento_reuniao': {'count': 1, 'percentage': 33.33}
# }
```

## 💻 Scripts de Treino

### train_naive_bayes.py

Script completo para treinar o classificador com cross-validation.

```bash
# Opções
python models/train_naive_bayes.py \
    --dataset dataset/dataset.json \
    --max-features 5000 \
    --ngrams 1 2 \
    --alpha 1.0 \
    --test-size 0.2 \
    --random-state 42 \
    --stopwords \
    --output-model models/naive_bayes_model.joblib \
    --output-vectorizer models/naive_bayes_vectorizer.joblib
```

**Funcionalidades:**
- Carregamento e validação de dataset
- Pré-processamento automático
- Train/test split com stratificação
- Treino e avaliação
- Cross-validation (5-fold)
- Feature importance
- Persistência automática

## 🔍 Scripts de Predição

### predict_naive_bayes.py

Script para fazer predições em novos emails.

```bash
# Texto individual
python models/predict_naive_bayes.py \
    --text "Conseguimos reunir amanhã?"

# Ficheiro JSON
python models/predict_naive_bayes.py \
    --file emails.json \
    --output predictions.json

# Modo interativo
python models/predict_naive_bayes.py --interactive

# Com top-N predições e detalhes
python models/predict_naive_bayes.py \
    --text "Email aqui" \
    --top-n 3 \
    --detailed
```

**Output esperado (JSON):**
```json
{
  "prediction": "agendamento_reuniao",
  "confidence": 0.9123,
  "top_predictions": [
    {"class": "agendamento_reuniao", "confidence": 0.9123},
    {"class": "nao_reuniao", "confidence": 0.0654},
    {"class": "cancelamento_reuniao", "confidence": 0.0223}
  ]
}
```

## 📊 Scripts de Avaliação

### evaluate_naive_bayes.py

Avaliação experimental completa com comparação de modelos.

```bash
# Avaliar apenas Naive Bayes
python models/evaluate_naive_bayes.py --dataset dataset/dataset.json

# Comparar com outros modelos
python models/evaluate_naive_bayes.py --compare-models

# Tuning de hiperparâmetros
python models/evaluate_naive_bayes.py --tuning

# Gerar relatório
python models/evaluate_naive_bayes.py --report evaluation_report.txt
```

**Modelos comparados:**
- Naive Bayes Multinomial
- Logistic Regression
- Decision Tree

**Métricas calculadas:**
- Accuracy
- Precision (macro e weighted)
- Recall (macro e weighted)
- F1-score (macro e weighted)
- Confusion Matrix
- Classification Report
- Cross-validation scores

## 📝 Exemplos de Uso

### Exemplo Completo: Treino + Predição

```python
from models.naive_bayes_classifier import NaiveBayesEmailClassifier
from models.utils import load_dataset, preprocess_texts, combine_text_fields
from sklearn.model_selection import train_test_split

# 1. Carregar dataset
emails = load_dataset('dataset/dataset.json')
texts = [combine_text_fields(e) for e in emails]
labels = [e['label'] for e in emails]

# 2. Pré-processar
texts = preprocess_texts(texts, lowercase=True)

# 3. Dividir
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

# 4. Criar e treinar
clf = NaiveBayesEmailClassifier()
clf.fit(X_train, y_train)

# 5. Avaliar
metrics = clf.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1-score: {metrics['f1_macro']:.4f}")

# 6. Guardar
clf.save('models/nb_model.joblib', 'models/nb_vectorizer.joblib')

# 7. Carregar e usar
clf_loaded = NaiveBayesEmailClassifier()
clf_loaded.load('models/nb_model.joblib', 'models/nb_vectorizer.joblib')

# 8. Predições
result = clf_loaded.predict("Conseguimos reunir amanhã?")
confidence = clf_loaded.predict_proba("Conseguimos reunir amanhã?")

print(f"Predição: {result}")
print(f"Confiança: {confidence[result]:.2%}")
```

### Executar Exemplos

```bash
python models/examples_naive_bayes.py
```

## 🔧 Configuração

### Hiperparâmetros Recomendados

| Parâmetro | Recomendado | Intervalo |
|-----------|-------------|-----------|
| max_features | 5000 | 1000-10000 |
| ngram_range | (1, 2) | (1,1) a (1,3) |
| alpha (Laplace) | 1.0 | 0.01-5.0 |
| min_df | 1 | 1-5 |
| max_df | 1.0 | 0.7-1.0 |

### Pré-processamento

O sistema suporta várias opções de pré-processamento:

```python
from models.utils import preprocess_text

text = preprocess_text(
    email_text,
    remove_signatures=True,        # Remove assinaturas
    remove_threads_history=True,   # Remove threads ("Re:")
    remove_punctuation=False,      # Mantém pontuação
    remove_stopwords=False,        # Mantém stopwords
    lowercase=True                 # Converte para minúsculas
)
```

## 📈 Resultados Esperados

Com a configuração padrão no dataset PT-PT:

```
Accuracy:              ~85-90%
Precision (macro):     ~80-85%
Recall (macro):        ~75-85%
F1-score (macro):      ~78-84%
F1-score (weighted):   ~85-90%
```

Variação depende do balanceamento e qualidade do dataset.

## ✅ Boas Práticas Implementadas

1. **UTF-8 Support**: Suporte completo para português europeu
2. **Type Hints**: Tipagem completa para melhor IDE support
3. **Logging**: Sistema de logging detalhado
4. **Error Handling**: Tratamento robusto de erros
5. **Modularidade**: Código separado em funções reutilizáveis
6. **Documentation**: Docstrings comprehensive
7. **Reproducibility**: Random state fixo
8. **Stratification**: Estratificação em train/test split
9. **Persistence**: Save/load automático
10. **Validation**: Validação de inputs

## 🐛 Troubleshooting

### Erro: "Modelo não foi treinado"

**Solução:** Executar `clf.fit(X_train, y_train)` antes de `clf.predict()`

### Accuracy baixa

**Possíveis causas:**
- Dataset pequeno ou desbalanceado
- Features insuficientes (aumentar `max_features`)
- Pré-processamento inadequado

**Soluções:**
- Aumentar dataset
- Ajustar `alpha` e `ngram_range`
- Revisar pré-processamento

### Ficheiros não encontrados

**Solução:** Verificar caminhos e executar scripts de diretório raiz

## 📚 Referências

- [Sklearn MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
- [TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Naive Bayes Theory](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

## 📄 Licença

Projeto académico - Mestrado em Dissertação

## 👥 Autor

NLP Research Team - Email Recognition PT-PT

## 📞 Suporte

Para problemas ou sugestões, consulte a documentação em `guides/`
