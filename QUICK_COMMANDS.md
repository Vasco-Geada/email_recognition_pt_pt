# 🚀 QUICK COMMANDS - CLASSIFICADOR NAIVE BAYES

## ✅ Tudo Pronto! Copie e Cole os Comandos Abaixo

### 1️⃣ TESTE RÁPIDO (Validar que tudo funciona)

```bash
cd email_recognition_pt_pt
python test_quick_naive_bayes.py
```

**Resultado esperado:** ✓ TODOS OS TESTES PASSARAM

---

### 2️⃣ TREINAR MODELO (Usar o dataset atual)

```bash
# Treino com configuração padrão
python models/train_naive_bayes.py --dataset dataset/dataset.json

# Treino com customização
python models/train_naive_bayes.py \
    --dataset dataset/dataset.json \
    --max-features 5000 \
    --ngrams 1 2 \
    --alpha 1.0 \
    --test-size 0.2 \
    --random-state 42 \
    --output-model models/naive_bayes_model.joblib \
    --output-vectorizer models/naive_bayes_vectorizer.joblib
```

---

### 3️⃣ FAZER PREDIÇÕES

#### Predição Simples (um email)

```bash
python models/predict_naive_bayes.py --text "Conseguimos reunir amanhã?"

python models/predict_naive_bayes.py --text "A reunião fica cancelada"

python models/predict_naive_bayes.py --text "Envio o relatório em anexo"
```

#### Predição com Detalhes (top-3 + probabilidades)

```bash
python models/predict_naive_bayes.py \
    --text "Conseguimos reunir amanhã?" \
    --top-n 3 \
    --detailed
```

#### Predição em Batch (múltiplos emails)

```bash
python models/predict_naive_bayes.py \
    --file dataset/dataset.json \
    --batch \
    --output predictions.json
```

#### Modo Interativo

```bash
python models/predict_naive_bayes.py --interactive
```

---

### 4️⃣ AVALIAR MODELO

#### Avaliação Básica

```bash
python models/evaluate_naive_bayes.py --dataset dataset/dataset.json
```

#### Comparar com Outros Modelos (NB vs LR vs DT)

```bash
python models/evaluate_naive_bayes.py --compare-models
```

#### Tuning de Hiperparâmetros

```bash
python models/evaluate_naive_bayes.py --tuning
```

#### Gerar Relatório Completo

```bash
python models/evaluate_naive_bayes.py \
    --compare-models \
    --tuning \
    --report evaluation_report.txt
```

---

### 5️⃣ VER EXEMPLOS PRÁTICOS

```bash
python models/examples_naive_bayes.py
```

Inclui exemplos de:
1. Treino básico
2. Predição simples
3. Predição batch
4. Feature importance
5. Save/Load
6. Distribuição de classes

---

## 📝 USO PROGRAMÁTICO (Em Python)

### Import e Carregar Modelo

```python
from models.naive_bayes_classifier import NaiveBayesEmailClassifier
from models.utils import combine_text_fields

# Carregar modelo pré-treinado
clf = NaiveBayesEmailClassifier()
clf.load('models/naive_bayes_model.joblib',
         'models/naive_bayes_vectorizer.joblib')
```

### Fazer Predição

```python
# Predição simples
email = "Conseguimos reunir amanhã?"
prediction = clf.predict(email)
print(f"Classe: {prediction}")  # agendamento_reuniao

# Com confiança
probabilities = clf.predict_proba(email)
confidence = probabilities[prediction]
print(f"Confiança: {confidence:.2%}")  # 73.19%

# Top-3 predições
sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
for classe, prob in sorted_probs[:3]:
    print(f"{classe}: {prob:.2%}")
```

### Treinar Novo Modelo

```python
from models.naive_bayes_classifier import NaiveBayesEmailClassifier
from models.utils import load_dataset, preprocess_texts, combine_text_fields
from sklearn.model_selection import train_test_split

# Carregar dados
emails = load_dataset('dataset/dataset.json')
texts = [combine_text_fields(e) for e in emails]
labels = [e['label'] for e in emails]

# Pré-processar
texts = preprocess_texts(texts)

# Dividir
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

# Treinar
clf = NaiveBayesEmailClassifier(max_features=5000, alpha=1.0)
clf.fit(X_train, y_train)

# Avaliar
metrics = clf.evaluate(X_test, y_test)

# Guardar
clf.save('models/custom_model.joblib', 'models/custom_vectorizer.joblib')
```

### Feature Importance

```python
# Palavras mais importantes por classe
features = clf.get_feature_importance(top_n=10)

for class_label, top_features in features.items():
    print(f"\nClasse: {class_label}")
    for feature, score in top_features[:5]:
        print(f"  {feature}: {score:.4f}")
```

---

## 🔗 INTEGRAÇÃO COM PIPELINE

### Com preprocessing.preprocess

```python
import preprocessing.preprocess as prep
from models.naive_bayes_classifier import NaiveBayesEmailClassifier

# Email original
email = {
    "subject": "Reunião amanhã",
    "body": "Conseguimos reunir?"
}

# Usar pipeline existente
processed = prep.preprocessEmail(email)

# Classificar
clf = NaiveBayesEmailClassifier()
clf.load('models/naive_bayes_model.joblib',
         'models/naive_bayes_vectorizer.joblib')

intent = clf.predict(processed['clean_body'])
print(f"Intent: {intent}")
```

### Predição em Batch de Ficheiro JSON

```python
from models.predict_naive_bayes import EmailPredictor

predictor = EmailPredictor(
    'models/naive_bayes_model.joblib',
    'models/naive_bayes_vectorizer.joblib'
)

# Fazer batch predictions
results = predictor.predict_from_json('dataset/dataset.json')

# Ver resultados
for result in results:
    print(f"Email: {result['email_id']}")
    print(f"  Predição: {result['prediction']}")
    print(f"  Confiança: {result['confidence']:.2%}")
```

---

## 🧪 TESTES

### Teste Rápido (todos os módulos)

```bash
python test_quick_naive_bayes.py
```

### Suite de Testes Completa

```bash
python models/test_naive_bayes.py
```

---

## 📊 EXEMPLOS DE OUTPUT

### Predição Individual

```
======================================================================
📧 Predição: agendamento_reuniao
📊 Confiança: 73.19%
======================================================================
```

### Predição com Top-3

```
======================================================================
📧 Predição: agendamento_reuniao
📊 Confiança: 73.19%

📈 Top Predições:
   - agendamento_reuniao: 73.19%
   - nao_reuniao: 18.44%
   - cancelamento_reuniao: 08.37%
======================================================================
```

### Feature Importance

```
Classe: agendamento_reuniao
  ✓ amanhã: -3.3344
  ✓ reunir: -3.5471
  ✓ consegues: -3.5471
  ✓ tarde: -3.5471

Classe: cancelamento_reuniao
  ✓ sexta fica: -3.3304
  ✓ reunião de: -3.3304
  ✓ fica cancelada: -3.3304
```

---

## 📚 FICHEIROS IMPORTANTES

| Ficheiro | Propósito |
|----------|-----------|
| `models/naive_bayes_classifier.py` | Classe principal do classificador |
| `models/train_naive_bayes.py` | Script de treino |
| `models/predict_naive_bayes.py` | Script de predição |
| `models/evaluate_naive_bayes.py` | Script de avaliação |
| `models/utils.py` | Funções auxiliares |
| `models/examples_naive_bayes.py` | Exemplos práticos |
| `models/README_NAIVE_BAYES.md` | Documentação completa |
| `guides/NAIVE_BAYES_INTEGRATION.md` | Guia de integração |

---

## 🎯 WORKFLOW TÍPICO

```bash
# 1. Validar sistema
python test_quick_naive_bayes.py

# 2. Treinar modelo (se necessário)
python models/train_naive_bayes.py --dataset dataset/dataset.json

# 3. Fazer predições
python models/predict_naive_bayes.py --text "Email aqui"

# 4. Avaliar performance
python models/evaluate_naive_bayes.py --compare-models

# 5. Ver exemplos
python models/examples_naive_bayes.py
```

---

## ⚡ ATALHOS

### Predição Rápida

```bash
# Copiar e cola direto no terminal:
python models/predict_naive_bayes.py --text "Conseguimos reunir amanhã às 15h?"
```

### Batch Prediction

```bash
python models/predict_naive_bayes.py --file dataset/dataset.json --batch --output results.json
```

### Comparação de Modelos

```bash
python models/evaluate_naive_bayes.py --compare-models
```

---

## 🐛 Troubleshooting Rápido

### Erro: ModuleNotFoundError

**Solução:** Executar do diretório raiz do projeto

```bash
cd email_recognition_pt_pt
python models/train_naive_bayes.py
```

### Erro: Modelo não encontrado

**Solução:** Treinar primeiro

```bash
python models/train_naive_bayes.py --dataset dataset/dataset.json
```

### Resultados inconsistentes

**Solução:** Usar random_state fixo

```python
clf = NaiveBayesEmailClassifier(random_state=42)
```

---

## 📞 CHEAT SHEET

```bash
# Teste rápido
python test_quick_naive_bayes.py

# Treino
python models/train_naive_bayes.py --dataset dataset/dataset.json

# Predição simples
python models/predict_naive_bayes.py --text "sua query aqui"

# Predição batch
python models/predict_naive_bayes.py --file dataset.json --batch

# Avaliação
python models/evaluate_naive_bayes.py --compare-models

# Exemplos
python models/examples_naive_bayes.py
```

---

**Status:** ✅ Sistema Completo e Pronto para Uso

Copie os comandos acima e execute diretamente no terminal!
