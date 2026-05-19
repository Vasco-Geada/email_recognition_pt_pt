"""
FINE_TUNING_GUIDE.md

Guia Completo de Fine-Tuning para Question Answering

Este documento explica como fazer fine-tuning de modelos QA (BERTimbau)
com dados específicos do projeto para melhorar a performance.


Project: Email Recognition PT-PT
Version: 1.0
"""

# Fine-Tuning Guide for QA Models

## Índice
1. [Conceitos](#conceitos)
2. [Preparação de Dados](#preparação-de-dados)
3. [Setup do Ambiente](#setup-do-ambiente)
4. [Fine-Tuning com HuggingFace Trainer](#fine-tuning-com-huggingface-trainer)
5. [Avaliação e Validação](#avaliação-e-validação)
6. [Comparação de Modelos](#comparação-de-modelos)
7. [Troubleshooting](#troubleshooting)

---

## Conceitos

### O que é Fine-Tuning?

Fine-tuning é o processo de adaptar um modelo pré-treinado (ex: BERTimbau)
com dados específicos do seu domínio/tarefa.

**Benefícios:**
- Melhor performance em tarefas específicas
- Menor quantidade de dados necessária vs treino do zero
- Treinamento mais rápido
- Melhor adaptação a português informal

**Trade-offs:**
- Requer GPU para treino eficiente
- Requer ~500-1000 exemplos de QA no mínimo
- Risco de overfitting em datasets pequenos

### Quando fazer Fine-Tuning?

✓ Fazer fine-tuning se:
- Tem >500 exemplos de QA
- Performance do modelo pré-treinado é insuficiente
- Tem muitos erros em categorias específicas
- Quer adaptar ao linguajar informal

✗ Não fazer (usar pré-treinado) se:
- Tem poucos dados (<200 exemplos)
- Performance atual é adequada
- Recursos computacionais limitados

---

## Preparação de Dados

### Formato SQuAD

O formato padrão para QA é SQuAD:

```json
{
  "version": "1.1",
  "data": [
    {
      "title": "Email Collection",
      "paragraphs": [
        {
          "context": "Boas Ana, podemos reunir sexta às 15h no Teams?",
          "qas": [
            {
              "question": "Quando é a reunião?",
              "id": "1_time_0",
              "answers": [
                {
                  "text": "sexta às 15h",
                  "answer_start": 32
                }
              ],
              "is_impossible": false
            }
          ]
        }
      ]
    }
  ]
}
```

### Gerar Dataset SQuAD do nosso formato

Usar nosso gerador (veja `qa_dataset_generator.py`):

```python
from qa.qa_dataset_generator import QADatasetGenerator

generator = QADatasetGenerator(include_question_variations=True)
generator.load_gold_annotations('gold_annotations/output/gold.json')
generator.save_dataset('qa/output', formats=['squad'])
```

### Dividir em Train/Test/Validation

Recomendação:
- **Train**: 70% (~350-700 exemplos)
- **Validation**: 15% (~75-150 exemplos)
- **Test**: 15% (~75-150 exemplos)

```python
from qa.qa_dataset_generator import QADataset

# Carregar
dataset = QADataset.load_json('qa/output/qa_dataset.json')

# Dividir
train_set, temp_set = dataset.split(train_ratio=0.7)
val_set, test_set = temp_set.split(train_ratio=0.5)

# Salvar
train_set.save_squad_format('qa/train_squad.json')
val_set.save_squad_format('qa/val_squad.json')
test_set.save_squad_format('qa/test_squad.json')
```

### Validação de Dataset

Verificar qualidade antes de treinar:

```python
# Verificar se answers estão no contexto
for example in examples:
    for answer in example.answers:
        if answer not in example.context:
            print(f"⚠ Answer '{answer}' não encontrada em contexto")
```

---

## Setup do Ambiente

### Instalação de Dependências

```bash
# Dependências base
pip install -r qa/requirements_qa.txt

# Dependências adicionais para fine-tuning
pip install datasets
pip install accelerate
pip install wandb  # Para logging (opcional)

# Se usar GPU (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verificar Setup

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

---

## Fine-Tuning com HuggingFace Trainer

### Script de Fine-Tuning

Criar ficheiro `qa/fine_tune_qa.py`:

```python
import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from datasets import load_dataset
import numpy as np

# Configuração
MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
TRAIN_DATASET = 'qa/train_squad.json'
VAL_DATASET = 'qa/val_squad.json'
OUTPUT_DIR = 'qa/models/bertimbau-finetuned'

def prepare_dataset(dataset_path):
    """Carrega e processa dataset."""
    datasets = load_dataset('json', data_files={
        'train': TRAIN_DATASET,
        'validation': VAL_DATASET
    })
    return datasets

def tokenize_function(examples, tokenizer, max_length=384):
    """Tokeniza exemplos."""
    questions = [q.strip() for q in examples["questions"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_index = i
        answers = examples["answers"][sample_index]
        
        if not answers["answer_start"]:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            
            sequence_ids = inputs.sequence_ids(i)
            
            # Encontrar tokens
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            
            token_end_index = len(sequence_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            
            token_end_index += 1
            
            if offset[token_start_index][0] <= start_char < offset[token_start_index][1]:
                pass
            else:
                token_start_index = -1
            
            if offset[token_end_index - 1][0] < end_char <= offset[token_end_index - 1][1]:
                pass
            else:
                token_end_index = -1
            
            start_positions.append(token_start_index)
            end_positions.append(token_end_index)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def compute_metrics(eval_pred):
    """Calcula métricas durante validação."""
    predictions, label_ids = eval_pred
    start_predictions = np.argmax(predictions[0], axis=1)
    end_predictions = np.argmax(predictions[1], axis=1)
    
    em = np.mean(
        (start_predictions == label_ids[0]) & 
        (end_predictions == label_ids[1])
    )
    
    return {"exact_match": em}

def main():
    # Carregar modelo e tokenizer
    print(f"Carregando modelo: {MODEL_NAME}")
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Preparar dataset
    print(f"Carregando datasets...")
    datasets = prepare_dataset(TRAIN_DATASET)
    
    # Tokenizar
    tokenized_datasets = datasets.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=datasets["train"].column_names
    )
    
    # Argumentos de treino
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        learning_rate=3e-5,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Treinar
    print("Iniciando treino...")
    trainer.train()
    
    # Salvar
    print(f"Salvando modelo em {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
```

### Executar Fine-Tuning

```bash
python qa/fine_tune_qa.py
```

---

## Avaliação e Validação

### Comparar Modelos

```python
from qa.qa_pipeline import QAPipeline
from qa.qa_evaluator import QAEvaluator

# Carregar dados de teste
test_data = load_test_data('qa/test_squad.json')

# Avaliar modelo original
print("\n=== Pre-trained Model ===")
pipeline_pretrained = QAPipeline(model_name='bertimbau-pt')
evaluator_pretrained = QAEvaluator()

for example in test_data:
    result = pipeline_pretrained.process_email(example['context'])
    evaluator_pretrained.evaluate_example(...)

evaluator_pretrained.print_report()

# Avaliar modelo fine-tuned
print("\n=== Fine-tuned Model ===")
pipeline_finetuned = QAPipeline(model_name='qa/models/bertimbau-finetuned')
evaluator_finetuned = QAEvaluator()

for example in test_data:
    result = pipeline_finetuned.process_email(example['context'])
    evaluator_finetuned.evaluate_example(...)

evaluator_finetuned.print_report()
```

### Analisar Melhorias

```python
# Comparar por categoria
categories = ['participants', 'time', 'location', 'topic']

for category in categories:
    pretrained_f1 = evaluator_pretrained.aggregated_metrics.per_category[category]['f1_score']
    finetuned_f1 = evaluator_finetuned.aggregated_metrics.per_category[category]['f1_score']
    
    improvement = ((finetuned_f1 - pretrained_f1) / pretrained_f1) * 100
    
    print(f"{category}: {improvement:+.1f}% improvement")
```

---

## Comparação de Modelos

### Configurar Experimentos

```python
import json
from typing import Dict, List

class ExperimentTracker:
    def __init__(self):
        self.experiments = []
    
    def add_result(
        self,
        model_name: str,
        dataset_name: str,
        exact_match: float,
        f1_score: float,
        inference_time: float,
        notes: str = ""
    ):
        self.experiments.append({
            'model': model_name,
            'dataset': dataset_name,
            'exact_match': exact_match,
            'f1_score': f1_score,
            'inference_time': inference_time,
            'notes': notes
        })
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.experiments, f, indent=2)
    
    def print_comparison(self):
        import pandas as pd
        df = pd.DataFrame(self.experiments)
        print(df.to_string())

# Usar
tracker = ExperimentTracker()

# Adicionar resultados
tracker.add_result(
    model_name='bertimbau-pt-pretrained',
    dataset_name='test_squad',
    exact_match=0.65,
    f1_score=0.75,
    inference_time=0.045,
    notes='Baseline pré-treinado'
)

tracker.add_result(
    model_name='bertimbau-pt-finetuned',
    dataset_name='test_squad',
    exact_match=0.72,
    f1_score=0.82,
    inference_time=0.048,
    notes='Finetuned por 3 épocas'
)

tracker.print_comparison()
tracker.save('qa/experiments/results.json')
```

---

## Troubleshooting

### Problema: Overfitting

**Sintomas:**
- Training loss desce, mas validation loss sobe
- EM/F1 em treino >> validation

**Soluções:**
```python
# 1. Reduzir number of epochs
num_train_epochs=2  # de 3

# 2. Aumentar regularização
weight_decay=0.05  # de 0.01

# 3. Usar dropout (alguns modelos)
model.config.hidden_dropout_prob = 0.2

# 4. Data augmentation - aumentar dataset
```

### Problema: Underfitting

**Sintomas:**
- Training loss não desce
- Baixa performance em treino e validação

**Soluções:**
```python
# 1. Aumentar learning rate
learning_rate=5e-5  # de 3e-5

# 2. Mais épocas
num_train_epochs=5  # de 3

# 3. Reduzir batch size
per_device_train_batch_size=8  # de 16

# 4. Verificar qualidade de dados
```

### Problema: Out of Memory (OOM)

**Soluções:**
```python
# 1. Reduzir batch size
per_device_train_batch_size=8  # de 16

# 2. Reduzir max_length
max_length=256  # de 384

# 3. Usar gradient accumulation
gradient_accumulation_steps=2

# 4. Usar modelo menor
MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'
# em vez de BERT-large
```

### Problema: Convergência Lenta

**Soluções:**
```python
# 1. Learning rate schedule
warmup_steps=100  # menos

# 2. Aumentar learning rate
learning_rate=5e-5  # de 3e-5

# 3. Usar otimizador mais avançado
optim="adamw_8bit"  # requer bitsandbytes
```

---

## Boas Práticas

### ✓ Faça

- Dividir dados em train/val/test
- Validar regularmente durante treino
- Guardar checkpoints
- Documentar hiperparâmetros
- Comparar com baseline
- Usar múltiplos seeds para reprodutibilidade

### ✗ Não Faça

- Treinar em GPU compartilhada sem limites
- Usar todo o dataset para treino (sem val/test)
- Alterar muitos hiperparâmetros de uma vez
- Treinar por muito tempo sem checkpoint
- Esquecer normalizar texto de entrada

---

## Recursos Adicionais

- [HuggingFace Course - Question Answering](https://huggingface.co/course/chapter7/2)
- [BERTimbau Paper](https://arxiv.org/abs/2009.06241)
- [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)

---

## Contato & Suporte

Implementado como parte do projeto "Email Recognition PT-PT"
Versão: 1.0
Última atualização: 2026-05-13
