# Arquitetura do Pipeline

## Fluxo Conceptual

Fluxo pretendido:

```text
Email raw
  -> anonymization
  -> preprocessing
  -> intent classification
  -> se intent == nao_reuniao: parar ou devolver saída mínima
  -> trigger extraction
  -> argument extraction
  -> temporal normalization
  -> structured output
```

## Componentes Existentes

### Anonymization

Ficheiros principais:

- `preprocessing/anonymization.py`
- `preprocessing/regex_anonymizer.py`
- `preprocessing/ner_anonymizer.py`
- `preprocessing/anonymization_config.py`
- `preprocessing/test_anonymization.py`
- `preprocessing/ANONYMIZATION_README.md`

Responsabilidades:

- substituir dados sensíveis antes de guardar/processar emails reais;
- preservar expressões temporais e plataformas úteis para reuniões;
- manter placeholders consistentes dentro do mesmo email;
- suportar modo `anonymize` por defeito e modo `pseudonymize` com mapping controlado.

Entidades cobertas:

- nomes de pessoas;
- emails;
- telefones;
- URLs;
- identificadores académicos/números de aluno;
- universidades e instituições de ensino;
- ORG/LOCAL via spaCy quando aplicável.

Whitelist preservada:

- Teams, Zoom, Discord, Moodle, GitHub, Google Meet, Outlook, Gmail, Slack, Trello, Notion.

Função central:

```python
anonymize_email(email: dict, keep_mapping: bool = False) -> dict
```

### Preprocessing

Ficheiros principais:

- `preprocessing/preprocess.py`
- `preprocessing/cleaning.py`
- `preprocessing/metadata.py`

Responsabilidades:

- normalizar texto;
- remover histórico de email;
- remover assinatura;
- dividir frases;
- tokenizar;
- extrair metadados como reply/thread level.

Função central:

```python
preprocessEmail(email)
```

Saída inclui `clean_body`, `sentences`, `tokens` e metadados.

### Intent Classification

Há dois caminhos principais:

- Logistic Regression + TF-IDF: `models/train_intent.py`, `models/predict_intent.py`.
- Naive Bayes + TF-IDF: `models/naive_bayes_classifier.py`, `models/train_naive_bayes.py`, `models/predict_naive_bayes.py`, `models/evaluate_naive_bayes.py`.

Artefactos guardados:

- `models/intent_classifier.joblib`
- `models/tfidf_vectorizer.joblib`
- `models/naive_bayes_model.joblib`
- `models/naive_bayes_vectorizer.joblib`

### Trigger Extraction

Ficheiro principal:

- `preprocessing/trigger_extraction.py`

Classe principal:

```python
TriggerExtractor
```

Abordagem:

- léxicos por intenção;
- matching exato;
- regex;
- lematização opcional com spaCy.

Ponto importante: a extração de trigger é intent-aware. O sistema primeiro usa a intenção prevista e só procura triggers relevantes para essa intenção.

### Argument Extraction

Ficheiro principal:

- `preprocessing/argument_extraction.py`

Classes:

- `ArgumentSpan`
- `ExtractedArguments`
- `TemporalExpressionExtractor`
- `LocationExtractor`
- `ParticipantExtractor`
- `TopicExtractor`
- `ArgumentExtractor`

Abordagem:

- regex para tempo e local;
- spaCy `pt_core_news_sm` para entidades e noun chunks;
- heurísticas para participantes;
- keywords para tópicos.

### Temporal Normalization

Ficheiro principal:

- `preprocessing/temporal_normalization.py`

Classe principal:

```python
TemporalNormalizer
```

Converte expressões como `amanhã`, `sexta às 15h`, `de tarde`, `para a semana` para estruturas com data/hora ISO, usando uma data de referência.

### Pipeline Integrado

Há dois pipelines:

- `preprocessing/email_pipeline.py`: intenção + trigger + prioridade/ação.
- `preprocessing/email_pipeline_enhanced.py`: recebe intenção prevista e extrai trigger + argumentos.

O pipeline enriquecido ainda não parece integrar explicitamente a normalização temporal na saída final. Ele extrai `time_expressions`, mas não converte automaticamente cada expressão com `TemporalNormalizer`.

## Saída Estruturada Pretendida

Exemplo conceptual:

```json
{
  "intent": "agendamento_reuniao",
  "intent_confidence": 0.92,
  "trigger": "agendar",
  "arguments": {
    "participants": ["Ana"],
    "time_expressions": ["sexta às 15h"],
    "normalized_time": [{"original_text": "sexta às 15h", "normalized_datetime": "..."}],
    "locations": ["Teams"],
    "topics": ["dataset"]
  }
}
```
