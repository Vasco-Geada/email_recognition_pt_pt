# Avaliação Experimental

## Avaliação de Intenção

Métricas usadas/documentadas:

- accuracy;
- precision;
- recall;
- F1 macro;
- F1 weighted;
- classification report;
- confusion matrix;
- cross-validation.

Ficheiros:

- `models/evaluate_naive_bayes.py`
- `models/train_intent.py`
- `models/test_naive_bayes.py`
- `test_quick_naive_bayes.py`
- `models/evaluate_decision_tree.py`

Risco principal: resultados atuais de 100% não são robustos se forem calculados em `dataset/dataset.json`, porque esse dataset tem apenas 16 exemplos e repetições.

Para o baseline Decision Tree, a avaliação foi configurada para usar por defeito `dataset/realistic_emails_v2.json`, não `dataset/dataset.json`.

Resultado observado em 22/05/2026:

- amostras: 300;
- classes: `agendamento_reuniao`, `cancelamento_reuniao`, `reuniao_confirmada`;
- hold-out estratificado 80/20: accuracy 1.0000, macro F1 1.0000, weighted F1 1.0000;
- 5-fold CV weighted F1: 0.9667 (+/- 0.0149).

Interpretação recomendada: usar como baseline interpretável, mas discutir possível facilidade do dataset/sinais lexicais e comparar no mesmo split com Naive Bayes, Logistic Regression e BERTimbau.

## Avaliação de Argumentos

Pasta:

- `evaluation/`

Ficheiros principais:

- `evaluation/evaluate_arguments.py`
- `evaluation/span_matching.py`
- `evaluation/metrics.py`
- `evaluation/report_generator.py`
- `evaluation/model_comparison.py`
- `evaluation/visualization.py`

Tipos de argumentos avaliados:

- `participants`
- `time`
- `location`
- `topic`

Estratégias de matching:

- exact match;
- partial match;
- fuzzy/token overlap;
- normalização textual configurável.

Saídas planeadas:

- JSON;
- CSV;
- Markdown;
- LaTeX;
- visualizações.

## Relatório Atual de Gold Annotations

Ficheiro observado:

- `gold_annotations/output/evaluation_report.json`

Estado observado:

- `total_samples`: 50;
- `exact_match_accuracy`: 1.0;
- precision/recall/F1: 1.0 para intenções e argumentos.

Cautela: este relatório parece comparar predições geradas pelo mesmo processo/heurística ou dados muito próximos das anotações. Não deve ser apresentado como avaliação final sem confirmar independência entre gold e predictions.

## Protocolo Recomendado para Tese

1. Definir dataset final com split fixo:
   - train;
   - dev/validation;
   - test.
2. Garantir distribuição por classe, incluindo `nao_reuniao` e `reuniao_confirmada`.
3. Criar gold manual ou semi-manual revisto.
4. Separar claramente:
   - pseudo-gold heurístico;
   - gold manual;
   - predictions do modelo.
5. Avaliar intenção separadamente da extração de argumentos.
6. Avaliar pipeline end-to-end com propagação de erro.
7. Reportar macro-F1 por classe e por argumento.
8. Incluir análise qualitativa de erros.
