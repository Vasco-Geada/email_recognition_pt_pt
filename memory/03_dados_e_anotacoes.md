# Dados e Anotações

## Dataset Principal

Ficheiro:

- `dataset/dataset.json`

Formato:

```json
[
  {
    "subject": "...",
    "body": "...",
    "label": "agendamento_reuniao"
  }
]
```

Estado observado:

- contém 16 exemplos;
- parece ter repetições dos mesmos 4 exemplos;
- inclui `agendamento_reuniao`, `cancelamento_reuniao` e `nao_reuniao`;
- não foi observado `reuniao_confirmada` neste ficheiro principal durante a leitura inicial.

Implicação: métricas de 100% neste dataset não devem ser tratadas como evidência forte, porque o dataset é pequeno e duplicado.

## Dataset Realista/Sintético

Ficheiro:

- `dataset/realistic_emails_v2.json`

Estado observado:

- contém 300 emails;
- inclui campos `subject`, `body`, `label`, `persona`;
- contém emails curtos, informais, com assinaturas/disclaimers, emojis e mistura PT/EN;
- parece mais adequado para experimentação inicial do que `dataset.json`.

Exemplos de personas observadas:

- `professor`
- `aluno_informal`
- `aluno_stressado`
- `aluno_internacional`

## Gold Annotations

Pasta:

- `gold_annotations/`

Ficheiros relevantes:

- `gold_annotations/gold_annotations_generator.py`
- `gold_annotations/heuristic_extractors.py`
- `gold_annotations/validators.py`
- `gold_annotations/evaluate_annotations.py`
- `gold_annotations/output/gold_annotations_v1.json`
- `gold_annotations/output/gold.json`
- `gold_annotations/output/predictions_v1.json`
- `gold_annotations/output/evaluation_report.json`

Formato das anotações:

```json
{
  "id": 1,
  "text": "...",
  "intent": "agendamento_reuniao",
  "trigger": ["reunir"],
  "arguments": {
    "participants": [],
    "time": [],
    "location": [],
    "topic": []
  },
  "confidence": {},
  "metadata": {}
}
```

## Cautela Crítica

As anotações `gold_annotations_v1.json` parecem ter sido geradas por heurísticas e validadas estruturalmente, não necessariamente revistas manualmente.

Foram observados exemplos suspeitos:

- `Cumprimentos`, `aula`, `telemóvel`, `vai`, `Bora`, `reunião` aparecem como participantes.
- Isto indica ruído relevante nas anotações de argumentos.

Conclusão: estes ficheiros são úteis como bootstrapping/pseudo-gold, mas para uma tese devem ser separados de gold manual real. Recomenda-se criar uma versão `gold_manual_v1.json` ou marcar explicitamente o estado de revisão por anotação.

