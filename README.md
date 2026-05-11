# Reconhecimento Automático de Eventos e Expressões Temporais em Emails: Desenvolvimento de um Modelo em Português Europeu
O email permanece um meio central de comunicação em contextos académicos, profissionais e organizacionais, sendo frequentemente utilizado para anunciar eventos, definir prazos, atribuir tarefas e coordenar atividades. Esta informação, apesar de relevante para a gestão do tempo e da produtividade, encontra-se expressa de forma não estruturada, recorrendo à linguagem natural, frequentemente informal e dependente do contexto comunicativo.

---

## 📁 Estrutura do Projeto

```
email_recognition_pt_pt/
├── README.md                                    # Este ficheiro
├── QUICKSTART.md                                # Guia de início rápido
├── requirements.txt                             # Dependências do projeto
├── emailExtraction.py                           # Script principal de extração de emails
├── emailTest.py                                 # Testes de emails
├── test_argument_extraction.py                  # Testes de extração de argumentos
│
├── models/                                      # Modelos treinados
│   ├── train_intent.py                         # Script de treino do classificador de intenção
│   ├── predict_intent.py                       # Script de predição de intenção
│   ├── intent_classifier.joblib                # Modelo treinado (Logistic Regression)
│   └── tfidf_vectorizer.joblib                 # Vetorizador TF-IDF treinado
│
├── preprocessing/                               # Módulo de pré-processamento
│   ├── __init__.py
│   ├── preprocess.py                           # Funções de pré-processamento gerais
│   ├── cleaning.py                             # Limpeza de texto
│   ├── metadata.py                             # Extração de metadados
│   ├── email_pipeline.py                       # Pipeline de processamento de emails
│   ├── email_pipeline_enhanced.py              # Pipeline melhorado
│   ├── trigger_extraction.py                   # Extração de triggers/expressões
│   ├── trigger_examples.py                     # Exemplos de triggers
│   ├── temporal_normalization.py               # Normalização de expressões temporais
│   └── argument_extraction.py                  # Extração de argumentos
│
├── argumentExtraction/                          # Módulo de extração de argumentos
│   └── old_code.py                             # Código antigo (referência)
│
├── temporalNormalization/                       # Módulo de normalização temporal
│   ├── examples_temporal_normalization.py      # Exemplos de normalização temporal
│   └── test_temporal_normalization.py          # Testes de normalização temporal
│
├── dataset/                                     # Dados de treino e testes
│   ├── dataset.json                            # Dataset principal (estrutura JSON)
│   ├── temp_emails.json                        # Dataset temporário de emails
│   └── generate_emails.py                      # Script para gerar dados de exemplo
│
└── guides/                                      # Documentação e guias
    ├── ARCHITECTURE.md                         # Descrição da arquitetura do projeto
    ├── IMPLEMENTATION_GUIDE.md                 # Guia de implementação
    ├── IMPLEMENTATION_SUMMARY_TEMPORAL.md      # Resumo de implementação temporal
    ├── ARGUMENT_EXTRACTION_GUIDE.md            # Guia de extração de argumentos
    ├── ARGUMENT_EXTRACTION_INDEX.md            # Índice de extração de argumentos
    ├── ARGUMENT_EXTRACTION_QUICKSTART.md       # Quickstart de extração de argumentos
    ├── ARGUMENT_EXTRACTION_SUMMARY.md          # Resumo de extração de argumentos
    ├── TEMPORAL_NORMALIZATION_GUIDE.md         # Guia de normalização temporal
    ├── TEMPORAL_NORMALIZATION_README.md        # README de normalização temporal
    └── TRIGGER_EXTRACTION_GUIDE.md             # Guia de extração de triggers
```

### Descrição dos Componentes

- **models/**: Contém o classificador de intenções de email (TF-IDF + Logistic Regression) e scripts de treino/predição
- **preprocessing/**: Pipeline de processamento de texto, limpeza, extração de features e normalização
- **argumentExtraction/**: Funcionalidades para extração de argumentos/entidades de emails
- **temporalNormalization/**: Módulo para reconhecer e normalizar expressões temporais em português
- **dataset/**: Dados de treino em formato JSON e utilitários de geração de dados
- **guides/**: Documentação técnica detalhada sobre cada componente

---

## 📧 Email Intent Classification Model

### Overview
A baseline text classification pipeline that classifies email intents using **TF-IDF vectorization** and **Logistic Regression**. The model is designed to classify Portuguese emails into one of four intent categories:

- **agendamento_reuniao**: Schedule/request a meeting
- **cancelamento_reuniao**: Cancel/decline a meeting
- **reuniao_confirmada**: Discuss/negotiate meeting date/time
- **nao_reuniao**: Not related to meetings

### Features
✅ Stratified 80/20 train/test split  
✅ TF-IDF vectorization (unigrams + bigrams, 5000 features)  
✅ Logistic Regression with balanced class weights  
✅ Full evaluation with classification report and confusion matrix  
✅ Portuguese accent preservation (no accent stripping)  
✅ Robust error handling and missing value management  
✅ Model and vectorizer persistence using joblib  

### Formato do dataset
O dataset é um ficheiro JSON que segue a seguinte estrutura:

```json
[
  {
    "subject": "Reunião com o cliente",
    "body": "Gostaria de agendar uma reunião para discutir o projeto.",
    "label": "agendamento_reuniao"
  },
  {
    "subject": "Cancelo a reunião",
    "body": "Infelizmente tenho que cancelar a reunião de amanhã.",
    "label": "cancelamento_reuniao"
  }
]
```

**Campos obrigatórios:**
- `subject` (str): Email subject line
- `body` (str): Email body/content
- `label` (str): Intent class (one of the four categories above)

### Installation

1. **Install required dependencies:**
   ```bash
   pip install scikit-learn pandas numpy joblib
   ```

2. **Prepare your dataset:**
   - Place your JSON dataset in `dataset/dataset.json`
   - Ensure the JSON structure matches the format above

### Running the Model

**Basic usage:**
```bash
python models/train_intent.py
```

This will:
1. Load the dataset from `dataset/dataset.json`
2. Preprocess and combine subject + body text
3. Split data into 80% training and 20% testing (stratified)
4. Vectorize text using TF-IDF
5. Train a Logistic Regression classifier
6. Evaluate on test set with detailed metrics
7. Save the trained model and vectorizer to `models/`

**Output:**
- Console logs showing progress and evaluation metrics
- Classification report with precision, recall, F1-score per class
- Confusion matrix showing prediction performance
- Saved files:
  - `models/intent_classifier.joblib` - Trained model
  - `models/tfidf_vectorizer.joblib` - TF-IDF vectorizer

### Model Configuration

Key hyperparameters (easily adjustable in the `main()` function):

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `test_size` | 0.2 | 20% of data for testing |
| `max_features` | 5000 | Maximum TF-IDF vocabulary size |
| `ngram_range` | (1, 2) | Unigrams and bigrams |
| `max_iter` | 1000 | Logistic Regression iterations |
| `class_weight` | "balanced" | Handle class imbalance |
| `random_state` | 42 | Reproducibility |

### Using the Trained Model

To load and use a saved model for predictions:

```python
from models.train_intent import EmailIntentClassifier

# Load the trained model
classifier = EmailIntentClassifier.load_model('models')

# Prepare new email text
new_email_subject = "Podemos marcar uma reunião?"
new_email_body = "Tenho disponibilidade na próxima terça."
email_text = f"{new_email_subject} {new_email_body}"

# Vectorize
X_new = classifier.vectorizer.transform([email_text])

# Predict
prediction = classifier.model.predict(X_new)
intent = classifier.class_labels[prediction[0]]
confidence = classifier.model.predict_proba(X_new).max()

print(f"Predicted Intent: {intent}")
print(f"Confidence: {confidence:.2%}")
```

### Evaluation Metrics

The model provides:
- **Accuracy**: Overall correctness on test set
- **Precision**: True positives / (True positives + False positives) per class
- **Recall**: True positives / (True positives + False negatives) per class
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Shows misclassification patterns

### Handling Missing Values

The script automatically:
- Detects missing/null values in subject, body, or label
- Removes rows with missing required fields
- Logs warnings when data is removed
- Continues safely with remaining data

### Portuguese Text Handling

✅ **Accent Preservation**: The TF-IDF vectorizer is configured with `strip_accents=None` to preserve Portuguese diacritics (ã, ç, é, etc.)  
This is crucial for maintaining semantic meaning in Portuguese text.

### Troubleshooting

**Issue**: `FileNotFoundError: Dataset file not found`
- **Solution**: Ensure `dataset/dataset.json` exists with proper JSON structure

**Issue**: `ValueError: Missing required columns`
- **Solution**: Check that JSON has exactly `subject`, `body`, and `label` fields

**Issue**: Class imbalance warning
- **Solution**: The model uses `class_weight='balanced'` to automatically handle imbalanced classes

**Issue**: Low model performance
- **Solution**: Check data quality, consider collecting more samples, adjust hyperparameters

### Code Structure

```
models/
├── train_intent.py          # Main training script
└── *.joblib                 # Saved model files (after training)

EmailIntentClassifier class includes:
├── load_dataset()           # Load and validate JSON data
├── prepare_data()           # 80/20 stratified split
├── vectorize_text()         # TF-IDF feature extraction
├── train_model()            # Logistic Regression training
├── evaluate_model()         # Comprehensive evaluation
└── save_model()             # Persist model and vectorizer
```

### References

- **Scikit-learn**: [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) | [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- **Joblib**: [Model persistence](https://joblib.readthedocs.io/en/latest/)
