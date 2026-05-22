# Modelos e Componentes

## Logistic Regression Baseline

Ficheiro:

- `models/train_intent.py`

Classe:

```python
EmailIntentClassifier
```

Características:

- TF-IDF com unigrams + bigrams;
- `max_features=5000`;
- preserva acentos com `strip_accents=None`;
- Logistic Regression com `class_weight='balanced'`;
- split estratificado 80/20;
- guarda modelo e vectorizer com joblib.

Nota: a lista `class_labels` no código observado inclui apenas:

- `agendamento_reuniao`
- `cancelamento_reuniao`
- `nao_reuniao`

Isto deve ser revisto se `reuniao_confirmada` for classe oficial.

## Naive Bayes Baseline

Ficheiros:

- `models/naive_bayes_classifier.py`
- `models/train_naive_bayes.py`
- `models/predict_naive_bayes.py`
- `models/evaluate_naive_bayes.py`
- `models/utils.py`

Classe:

```python
NaiveBayesEmailClassifier
```

Características:

- TF-IDF;
- MultinomialNB;
- suporte a save/load;
- `predict`, `predict_proba`, `evaluate`;
- análise de feature importance;
- comparação experimental com Logistic Regression e Decision Tree em `evaluate_naive_bayes.py`.

Estado documentado:

- relatórios atuais indicam 100% em dataset pequeno;
- esse resultado é provavelmente inflacionado por dataset reduzido/duplicado.

## Decision Tree Baseline

Ficheiros:

- `models/decision_tree_classifier.py`
- `models/train_decision_tree.py`
- `models/predict_decision_tree.py`
- `models/evaluate_decision_tree.py`
- `models/README_DECISION_TREE.md`

Classe:

```python
DecisionTreeEmailClassifier
```

Dataset obrigatório por defeito:

- `dataset/realistic_emails_v2.json`

Características:

- TF-IDF com `lowercase=True`, `strip_accents=None`, `ngram_range=(1, 2)`, `max_features=5000`;
- `DecisionTreeClassifier`;
- split estratificado 80/20;
- hiperparâmetros configuráveis: `max_depth`, `min_samples_split`, `min_samples_leaf`;
- persistência em `models/decision_tree_model.joblib` e `models/decision_tree_vectorizer.joblib`;
- inferência com classe prevista, probabilidades por classe e confiança;
- baseline interpretável para comparar com Naive Bayes, Logistic Regression e BERTimbau.

Estado validado em 22/05/2026:

- dataset carregado: 300 emails;
- distribuição: 120 `agendamento_reuniao`, 90 `cancelamento_reuniao`, 90 `reuniao_confirmada`;
- split 80/20: 240 treino, 60 teste;
- teste hold-out: accuracy/macro F1/weighted F1 = 1.0000;
- 5-fold CV weighted F1 = 0.9667 (+/- 0.0149).

Nota: estes resultados são bons, mas devem ser interpretados com cautela se o dataset for sintético ou tiver padrões lexicais muito fortes.

## Extração de Argumentos

Ficheiro:

- `preprocessing/argument_extraction.py`

Saída base:

```python
ExtractedArguments(
    participants=[ArgumentSpan(...)]
    time_expressions=[ArgumentSpan(...)]
    locations=[ArgumentSpan(...)]
    topics=[ArgumentSpan(...)]
)
```

Pontos fortes:

- modular;
- spans têm offset, confiança e método;
- combina regex, NER e heurísticas;
- interpretável.

Pontos frágeis:

- spaCy PT genérico pode falhar em emails informais;
- heurísticas de participantes produzem falsos positivos;
- tópicos por noun chunks/keywords são ruidosos;
- algumas regex parecem ter encoding corrompido nos literais quando vistas na consola.

## QA Module

Pasta:

- `qa/`

Objetivo:

- usar Question Answering com transformers para extrair participantes, tempo, local e tópico.

Modelos documentados:

- `neuralmind/bert-base-portuguese-cased`
- `bert-base-multilingual-cased`

Estado:

- há documentação, pipeline, geração de dataset QA, avaliação e guias de fine-tuning;
- provavelmente é um caminho alternativo/avançado para comparar contra heurísticas.

## Gold Annotation Tools

Pasta:

- `gold_annotations/`

Objetivo:

- gerar anotações iniciais com heurísticas;
- validar estrutura;
- avaliar predições contra anotações.

Uso recomendado:

- usar como ferramenta de bootstrap;
- não chamar de gold final até haver revisão manual.
