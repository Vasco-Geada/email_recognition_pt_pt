# Executar o projeto

## 1. Importar e anonimizar emails

As credenciais IMAP são lidas de `EMAIL`, `PASSWORD` e `SERVER` no `.env`.

```powershell
python emailExtraction.py
```

O resultado é guardado por defeito em:

```text
dataset/imported_emails_anonymized.json
```

Assunto, corpo, remetente, destinatários e participantes são anonimizados em
memória antes da escrita. O ficheiro não contém o texto original nem o mapa de
substituições.

## 2. Treinar ou atualizar os classificadores

```powershell
python run_classification_models.py train `
  --dataset dataset/dataset_treino_anonimizado.json
```

Os novos modelos substituem os anteriores em `trained_models/email_intent`.

## 3. Executar o projeto com modelos persistidos

```powershell
python run_project.py dataset/imported_emails_anonymized.json
```

Este comando não treina nem anonimiza. Executa:

- Logistic Regression, Naive Bayes e Decision Tree;
- extração clássica de participantes, tempo, localização e tópico;
- QA com o BERTimbau fine-tuned.

A Logistic Regression fornece a intenção principal aos extratores. As
previsões dos três modelos continuam disponíveis nos resultados.

Os resultados são guardados automaticamente em
`evaluation_results/project_runs/<nome-do-dataset>`.
