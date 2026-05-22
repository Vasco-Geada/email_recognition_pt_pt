# Memória do Projeto

Esta pasta resume o estado do projeto `email_recognition_pt_pt` para retomar trabalho em novas sessões, com Codex ou outra ferramenta.

O projeto é uma tese/pipeline NLP em PT-PT para analisar emails e identificar eventos de reunião. O objetivo operacional é:

1. Extrair emails.
2. Classificar a intenção do email.
3. Se o email for sobre reunião, extrair trigger, argumentos e expressões temporais.
4. Se não for sobre reunião, classificar como `nao_reuniao`.

Classes de intenção atualmente usadas:

- `agendamento_reuniao`
- `cancelamento_reuniao`
- `reuniao_confirmada`
- `nao_reuniao`

Ficheiros desta memória:

- [01_contexto_tese.md](01_contexto_tese.md): objetivo, domínio, labels e definições.
- [02_arquitetura_pipeline.md](02_arquitetura_pipeline.md): componentes e fluxo end-to-end.
- [03_dados_e_anotacoes.md](03_dados_e_anotacoes.md): datasets, gold annotations e limitações.
- [04_modelos_e_componentes.md](04_modelos_e_componentes.md): modelos e módulos implementados.
- [05_avaliacao_experimental.md](05_avaliacao_experimental.md): avaliação, métricas e cautelas.
- [06_estado_atual_e_riscos.md](06_estado_atual_e_riscos.md): estado real, riscos técnicos e científicos.
- [07_proximos_passos.md](07_proximos_passos.md): backlog recomendado.

Leitura recomendada ao iniciar uma nova sessão: ler este ficheiro, depois `01`, `02`, `06` e `07`.

