# Estado Atual e Riscos

## Estado Atual

O projeto já contém uma base funcional para:

- preprocessing de emails;
- classificação de intenção com modelos clássicos;
- extração de triggers por léxico/regex;
- extração de argumentos por regex, spaCy e heurísticas;
- normalização temporal rule-based;
- geração de pseudo-gold annotations;
- avaliação de argumentos com várias métricas;
- módulo QA baseado em transformers como alternativa experimental.

## Principais Riscos Técnicos

### 1. Encoding/Mojibake

Vários ficheiros aparecem na consola com texto como `reuniÃ£o` em vez de `reunião`. Pode ser apenas a consola PowerShell, mas convém verificar encoding real dos ficheiros e garantir UTF-8 consistente.

Impacto:

- regex com acentos podem falhar;
- documentação fica ilegível;
- modelos podem aprender formas corrompidas.

### 2. Dataset Pequeno e Duplicado

`dataset/dataset.json` tem 16 exemplos e repetições.

Impacto:

- métricas inflacionadas;
- overfitting;
- conclusões frágeis.

### 3. `reuniao_confirmada` Nem Sempre Está Integrada

A classe é mencionada no objetivo e na documentação, mas no `train_intent.py` observado a lista base tinha apenas três classes, sem `reuniao_confirmada`.

Impacto:

- inconsistência experimental;
- classe oficial pode ficar fora do treino/avaliação.

### 4. Pseudo-Gold Ruidoso

As gold annotations geradas heuristicamente contêm falsos positivos visíveis em participantes.

Impacto:

- avaliação de argumentos fica enviesada;
- modelo pode ser avaliado contra erros da heurística.

### 5. Inconsistência de Schemas

Há diferentes nomes para os mesmos campos:

- `time`, `time_expressions`;
- `topic`, `topics`, `meeting_topics`;
- `location`, `locations`.

Impacto:

- integração frágil;
- avaliação pode ignorar campos;
- maior risco de bugs silenciosos.

### 6. Normalização Temporal Não Integrada End-to-End

O extrator encontra expressões temporais e o normalizador existe, mas o pipeline enriquecido ainda não parece devolver normalizações temporais automaticamente.

Impacto:

- requisito de normalização temporal fica incompleto na saída final.

## Riscos Científicos

- Confundir bootstrapping heurístico com anotação gold.
- Reportar métricas em dados gerados pelo mesmo sistema que produziu as anotações.
- Não avaliar emails `nao_reuniao` no pipeline end-to-end.
- Não separar avaliação de intenção, extração e normalização.
- Não documentar ambiguidade temporal com data de referência.

