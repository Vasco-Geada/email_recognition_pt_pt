# Contexto da Tese

## Objetivo

O projeto desenvolve um pipeline NLP para emails em português europeu, com foco no reconhecimento de reuniões. O sistema deve transformar texto de email não estruturado numa representação estruturada que indique:

- se o email é ou não sobre uma reunião;
- o tipo de ato comunicativo associado à reunião;
- o trigger que sinaliza esse ato;
- participantes, tempo, local e tópico;
- normalização temporal quando existirem expressões temporais.

## Problema Central

Emails académicos/profissionais em PT-PT são curtos, informais, dependentes de contexto e frequentemente contêm:

- respostas encadeadas;
- assinaturas;
- disclaimers;
- expressões vagas: `amanhã`, `sexta`, `depois da aula`, `quinta de manhã`;
- mistura de português e inglês: `quick call`, `meeting`, `Teams`;
- linguagem informal e emojis.

O sistema deve distinguir emails de reunião de emails não relacionados. Quando não for reunião, a saída esperada de intenção é `nao_reuniao`.

## Labels de Intenção

- `agendamento_reuniao`: pedido/proposta/negociação para marcar reunião.
- `cancelamento_reuniao`: cancelamento, indisponibilidade, adiamento ou impossibilidade.
- `reuniao_confirmada`: confirmação de presença, aceitação ou reunião já fixada.
- `nao_reuniao`: email sem evento de reunião.

Nota importante: algumas docs antigas ainda descrevem `reuniao_confirmada` como discussão/negociação de data/hora. Para a tese convém estabilizar uma definição única e alinhar dataset, anotações e relatório.

## Argumentos Extraídos

Tipos de argumentos usados no projeto:

- `participants`: pessoas, destinatários, nomes próprios ou emails.
- `time` / `time_expressions`: datas, horas, dias relativos e expressões temporais vagas.
- `location` / `locations`: sala, edifício, Teams, Zoom, campus, etc.
- `topic` / `topics` / `meeting_topics`: assunto da reunião.

Existe alguma inconsistência nominal entre módulos: `time` vs `time_expressions`, `topic` vs `topics` vs `meeting_topics`. Isto deve ser normalizado antes de uma avaliação final.

