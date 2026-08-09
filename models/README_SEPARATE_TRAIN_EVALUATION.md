# Treino e avaliação com datasets separados

O comando `train` usa todos os emails do dataset indicado para ajustar o
TF-IDF e os três classificadores. Não cria uma divisão interna de teste.

```powershell
python run_classification_models.py train `
  --dataset dataset/dataset_treino.json
```

São guardados o modelo e o vectorizer de:

- Logistic Regression
- Naive Bayes
- Decision Tree

O ficheiro `training_metadata.json` regista as classes, configuração de
pré-processamento, hash do dataset e fingerprints dos emails de treino.

Para avaliar modelos já treinados num segundo dataset:

```powershell
python run_classification_models.py evaluate `
  --dataset dataset/dataset_teste.json `
  --model-dir trained_models/email_intent `
  --output-dir evaluation_results/independent_classification
```

O comando `evaluate` nunca executa `fit()` nem altera os vectorizers. O
pré-processamento usado no treino é recuperado automaticamente dos metadados.
Os datasets importados já devem estar anonimizados; esta etapa não volta a
aplicar anonimização.

Por defeito, a avaliação é interrompida quando:

- o ficheiro de teste é o mesmo ficheiro usado no treino;
- existem emails processados iguais nos dois datasets;
- o teste contém uma classe que o modelo nunca observou no treino.

`--allow-overlap` existe apenas para experiências deliberadas e não deve ser
usado para reportar métricas de generalização.

Os resultados incluem, por modelo:

- previsões e probabilidades por email em JSON;
- accuracy, precision, recall e F1;
- matriz de confusão;
- análise de erros;
- resumo comparativo em CSV e JSON.
