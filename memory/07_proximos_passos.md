# Próximos Passos Recomendados

## Prioridade Alta

1. Normalizar o schema de saída.

   Escolher uma convenção única, por exemplo:

   ```json
   {
     "intent": "...",
     "trigger": "...",
     "arguments": {
       "participants": [],
       "time": [],
       "location": [],
       "topic": []
     },
     "temporal_normalization": []
   }
   ```

2. Garantir que `reuniao_confirmada` está presente em todos os pontos:

   - dataset;
   - treino;
   - predição;
   - avaliação;
   - trigger lexicons;
   - documentação.

3. Criar ou rever manualmente um gold set.

   Recomendação:

   - selecionar subconjunto de `realistic_emails_v2.json`;
   - rever intenção, trigger e argumentos;
   - marcar `metadata.validated = true`;
   - guardar como `gold_annotations/output/gold_manual_v1.json`.

4. Integrar normalização temporal no pipeline enriquecido.

   Para cada `time_expression`, chamar `TemporalNormalizer.normalize(...)` com uma `reference_datetime` explícita.

5. Reavaliar modelos com dataset maior e sem duplicados.

## Prioridade Média

1. Criar script único de pipeline end-to-end.

   Entrada:

   ```json
   {"subject": "...", "body": "...", "email_date": "..."}
   ```

   Saída:

   ```json
   {
     "intent": "...",
     "trigger": "...",
     "arguments": {},
     "normalized_temporals": []
   }
   ```

2. Adicionar testes para:

   - classificação de `nao_reuniao`;
   - emails sem reunião não passarem para extração de reunião;
   - `reuniao_confirmada`;
   - normalização de `amanhã`, `sexta às 15h`, `depois de almoço`.

3. Melhorar limpeza de assinatura/disclaimer antes de participantes.

4. Criar relatório experimental reprodutível:

   - comandos;
   - seed;
   - dataset usado;
   - métricas;
   - outputs.

## Prioridade Baixa

1. Comparar heurísticas de argumentos com QA/transformers.
2. Explorar BERTimbau para intent classification.
3. Criar análise de erros por persona.
4. Criar visualizações para a dissertação.

## Comandos Úteis

Treinar Naive Bayes:

```bash
python models/train_naive_bayes.py --dataset dataset/dataset.json
```

Avaliar Naive Bayes:

```bash
python models/evaluate_naive_bayes.py --compare-models --tuning
```

Treinar Logistic Regression:

```bash
python models/train_intent.py
```

Testes principais:

```bash
python test_quick_naive_bayes.py
python test_argument_extraction.py
python temporalNormalization/test_temporal_normalization.py
python gold_annotations/test_gold_annotations.py
```

