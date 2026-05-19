# Question Answering Module - Email Recognition PT-PT

Módulo completo de Question Answering para extração estruturada de informações em emails académicos informais em português europeu.

## 📋 Índice

- [Overview](#overview)
- [Características](#características)
- [Instalação](#instalação)
- [Quick Start](#quick-start)
- [Uso Avançado](#uso-avançado)
- [Arquitetura](#arquitetura)
- [Avaliação](#avaliação)
- [Fine-Tuning](#fine-tuning)
- [Exemplos](#exemplos)
- [FAQ](#faq)

---

## Overview

Este módulo implementa um pipeline de Question Answering (QA) baseado em transformers para extrair automaticamente informações estruturadas sobre reuniões em emails.

### Caso de Uso

```
Input Email:
"Boas Ana, podemos reunir sexta às 15h no Teams para discutir o dataset?"

Output:
{
  "participants": "Ana",
  "time": "sexta às 15h",
  "location": "Teams",
  "topic": "dataset"
}
```

### Modelos Suportados

- **Recomendado**: `neuralmind/bert-base-portuguese-cased` (BERTimbau)
- **Fallback**: `bert-base-multilingual-cased`
- **Custom**: Qualquer modelo QA de HuggingFace

---

## Características

### ✨ Principais

- ✅ **Português Nativo**: Otimizado para português europeu informal
- ✅ **4 Categorias Estruturadas**: Participantes, Hora, Local, Tópico
- ✅ **Perguntas Variadas**: Múltiplas variações de perguntas para robustez
- ✅ **Confiança Calibrada**: Scores de confiança com thresholding
- ✅ **Pós-processamento**: Limpeza automática de respostas
- ✅ **Avaliação Integrada**: Métricas EM, F1, Precision, Recall
- ✅ **Batch Processing**: Processamento eficiente de múltiplos emails
- ✅ **Caching**: Cache de resultados para performance
- ✅ **Fallback Multilíngue**: Suporte automático a modelo multilíngue se necessário

### 📊 Métricas

- **Exact Match (EM)**: Resposta completamente correta
- **F1 Score**: Overlap de tokens (Precision × Recall)
- **Confidence**: Score normalizado do modelo (0-1)
- **Error Analysis**: Classificação automática de erros

### 🔧 Componentes

```
qa/
├── qa_questions.py              # Perguntas estruturadas
├── qa_utils.py                  # Utilidades (limpeza, métricas)
├── qa_dataset_generator.py      # Geração de datasets
├── qa_inference.py              # Motor de inferência QA
├── qa_evaluator.py              # Avaliador de resultados
├── qa_pipeline.py               # Pipeline integrado
├── __init__.py                  # Imports
├── example_qa_usage.py          # Exemplos de uso
├── requirements_qa.txt          # Dependências
├── FINE_TUNING_GUIDE.md         # Guia de fine-tuning
└── README.md                    # Este ficheiro
```

---

## Instalação

### 1. Dependências Básicas

```bash
# Instalar requirements
pip install -r qa/requirements_qa.txt

# Ou instalação manual
pip install torch transformers numpy scikit-learn pandas
```

### 2. CUDA (GPU - Recomendado)

```bash
# Para CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. CPU Only

```bash
# Se GPU não disponível
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. Verificar Setup

```python
import torch
from transformers import AutoTokenizer

print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.is_available()}")

# Testar carregamento de modelo
tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
print("✓ Setup completo!")
```

---

## Quick Start

### Exemplo Básico

```python
from qa.qa_pipeline import QAPipeline

# Inicializar
pipeline = QAPipeline(
    model_name='bertimbau-pt',  # ou 'multilingual'
    confidence_threshold=0.5
)

# Processar email
email = "Boas Ana, reunimos sexta às 15h no Teams para discutir o dataset?"

result = pipeline.process_email(email_text=email)

# Obter respostas
answers = result.get_answers_only()
print(answers)
# {
#   'participants': 'Ana',
#   'time': 'sexta às 15h',
#   'location': 'Teams',
#   'topic': 'dataset'
# }
```

### Interface Ultra-Rápida

```python
from qa.qa_pipeline import QuickQA

# Inicializar (primeira vez)
QuickQA.init(model_name='multilingual', device='cpu')

# Usar
answers = QuickQA.answer("Reunimos segunda às 10h na sala 201?")
print(answers)
```

---

## Uso Avançado

### Batch Processing

```python
emails = [
    {'id': 1, 'subject': 'Meeting 1', 'text': 'Email 1...'},
    {'id': 2, 'subject': 'Meeting 2', 'text': 'Email 2...'},
]

results = pipeline.process_batch(emails, show_progress=True)

# Salvar resultados
pipeline.save_results(
    results,
    output_dir='qa/output',
    formats=['json', 'csv']
)
```

### Integração com Gold Annotations

```python
# Carregar anotações
gold_annotations = pipeline.load_gold_annotations(
    'gold_annotations/output/gold.json'
)

# Processar e integrar
integrated = pipeline.integrate_with_gold_annotations(
    gold_annotations,
    output_file='qa/output/integrated.json'
)
```

### Controlar Confiança

```python
# Threshold baixo (mais respostas, menos confiáveis)
pipeline_loose = QAPipeline(confidence_threshold=0.3)

# Threshold alto (menos respostas, mais confiáveis)
pipeline_strict = QAPipeline(confidence_threshold=0.8)

# Resultado com None se confiança baixa
result = pipeline_strict.process_email(email)
```

### Processamento com Cache

```python
# Cache automático
pipeline = QAPipeline(use_cache=True, cache_size=1000)

# Mesma pergunta/contexto reutiliza cache
result1 = pipeline.process_email(email)
result2 = pipeline.process_email(email)  # Rápido!
```

---

## Arquitetura

### Pipeline de Processamento

```
Email
  ↓
[Normalização de texto]
  ↓
[Gerar perguntas estruturadas]
  ↓
[Inferência QA (BERT)]
  ↓
[Pós-processamento]
  ↓
[Aplicar threshold de confiança]
  ↓
Respostas Estruturadas
```

### Componentes Principais

#### 1. **qa_questions.py**
Define perguntas fixas e variações em português:

```python
from qa.qa_questions import QAQuestions

# Obter pergunta primária
question = QAQuestions.get_primary_question(QuestionCategory.TIME)
# "Quando é a reunião?"

# Obter variação aleatória
variation = QAQuestions.get_random_variation(QuestionCategory.TIME)
# "A que horas é a reunião?"
```

#### 2. **qa_inference.py**
Motor de QA com transformers:

```python
from qa.qa_inference import QAInferenceEngine

engine = QAInferenceEngine(model_name='bertimbau-pt')
result = engine.answer_question(
    question="Quando?",
    context="Reunimos sexta"
)
print(result.answer)  # "sexta"
print(result.confidence)  # 0.85
```

#### 3. **qa_utils.py**
Utilidades de processamento:

```python
from qa.qa_utils import TextNormalizer, AnswerPostProcessor

# Limpeza de texto
clean = TextNormalizer.clean_text("  Olá   Ana  ")
# "Olá Ana"

# Filtrar artefatos
filtered = AnswerPostProcessor.filter_common_artifacts(
    "Ana 😅 DISCLAIMER: mensagem automática"
)
# "Ana"
```

#### 4. **qa_evaluator.py**
Métricas de avaliação:

```python
from qa.qa_evaluator import QAEvaluator

evaluator = QAEvaluator()

evaluator.evaluate_example(
    example_id='1',
    question='Quando?',
    predicted='sexta',
    reference='sexta às 15h',
    confidence=0.8
)

evaluator.print_report()
```

#### 5. **qa_pipeline.py**
Pipeline integrado:

```python
from qa.qa_pipeline import QAPipeline

pipeline = QAPipeline()
result = pipeline.process_email(email_text)
```

---

## Avaliação

### Usar Evaluador

```python
from qa.qa_evaluator import QAEvaluator

evaluator = QAEvaluator()

# Avaliar exemplos individuais
for pred in predictions:
    for ref in references:
        evaluator.evaluate_example(
            example_id=pred['id'],
            question=pred['question'],
            predicted=pred['answer'],
            reference=ref['answer'],
            confidence=pred['confidence'],
            category=pred['category']
        )

# Gerar relatório
evaluator.aggregate_metrics()
evaluator.print_report()
evaluator.print_error_analysis(top_n=5)

# Salvar
evaluator.save_results('qa/output/eval.json')
```

### Métricas Comuns

| Métrica | Fórmula | Interpretação |
|---------|---------|-----------------|
| **EM** | # Correct / Total | Percentagem de respostas exatas |
| **F1** | 2×(P×R)/(P+R) | Média harmónica precision/recall |
| **Precision** | # Correct Tokens / # Predicted | Quantos tokens estão certos |
| **Recall** | # Correct Tokens / # Reference | Quantos tokens de ref foram preditos |

### Exemplo de Relatório

```
======================================================================
QA EVALUATION REPORT
======================================================================

Total Examples: 100

Global Metrics:
  Exact Match (EM): 0.7200
  Mean F1 Score: 0.8150
  Mean Confidence: 0.7890

Metrics by Category:
  participants:
    EM: 0.6500
    F1: 0.7800
    Count: 25
  time:
    EM: 0.8000
    F1: 0.8900
    Count: 25
  ...

Error Distribution:
  EXACT_MATCH: 72 (72.0%)
  PARTIAL_MATCH: 18 (18.0%)
  WRONG_ANSWER: 10 (10.0%)
  ...
======================================================================
```

---

## Fine-Tuning

Para melhor performance com dados específicos (requer >500 exemplos):

### Preparar Dataset

```python
from qa.qa_dataset_generator import QADatasetGenerator

generator = QADatasetGenerator()
generator.load_gold_annotations('gold_annotations/output/gold.json')
generator.save_dataset('qa/output', formats=['squad'])
```

### Fine-Tune Modelo

Ver [FINE_TUNING_GUIDE.md](FINE_TUNING_GUIDE.md) para instruções completas.

```bash
python qa/fine_tune_qa.py
```

### Usar Modelo Fine-Tuned

```python
pipeline = QAPipeline(
    model_name='qa/models/bertimbau-finetuned'
)
```

---

## Exemplos

### Exemplo 1: Uso Simples

```python
from qa import QuickQA

QuickQA.init()
answers = QuickQA.answer("Boas, reunimos amanhã à tarde?")
print(answers)
```

### Exemplo 2: Batch com Salvar

```python
from qa import QAPipeline

pipeline = QAPipeline()

emails = [
    {'id': 1, 'text': 'Email 1...'},
    {'id': 2, 'text': 'Email 2...'},
]

results = pipeline.process_batch(emails)
pipeline.save_results(results, 'output/', formats=['json', 'csv'])
```

### Exemplo 3: Comparação Métodos

```python
from qa import QAEvaluator

# Avaliar com threshold baixo
evaluator_low = QAEvaluator()
# ... avaliar ...

# Avaliar com threshold alto
evaluator_high = QAEvaluator()
# ... avaliar ...

print("Low confidence threshold:")
evaluator_low.print_report()

print("\nHigh confidence threshold:")
evaluator_high.print_report()
```

Ver [example_qa_usage.py](example_qa_usage.py) para 7 exemplos completos!

```bash
python qa/example_qa_usage.py
```

---

## FAQ

### P: Qual modelo usar?

**R:** Recomendação de ordem:
1. `neuralmind/bert-base-portuguese-cased` (melhor para PT)
2. `neuralmind/bert-large-portuguese-cased` (mais lento, maior)
3. `bert-base-multilingual-cased` (fallback)

### P: Preciso GPU?

**R:** Não é obrigatório, mas recomendado:
- **CPU**: ~0.5-1s por email
- **GPU**: ~0.05-0.1s por email

### P: Quantos exemplos para fine-tuning?

**R:** Mínimo 300-500 exemplos QA, ideal 1000+.

### P: Como melhorar performance?

**R:** Em ordem de impacto:
1. Usar BERTimbau em vez de multilíngue
2. Fine-tuning com dados do projeto
3. Aumentar confidence_threshold (menos respostas)
4. Usar variações de perguntas

### P: Posso usar sem internet?

**R:** Primeira vez requer download de modelo (~400MB).
Depois funciona offline.

### P: Que ambientes são suportados?

**R:** Python 3.8+, funciona em:
- Windows 10/11
- macOS 10.14+
- Linux (Ubuntu 18.04+)

### P: Há suporte para outros idiomas?

**R:** Sim, com `bert-base-multilingual-cased`, mas performance é inferior.

### P: Performance em emails muito longos?

**R:** Modelo BERT limita a ~512 tokens. Emails muito longos são truncados.
Solução: Usar trecho relevante ou fine-tune com max_length ajustado.

---

## Estrutura de Diretórios

```
qa/
├── qa_questions.py              # Definições de perguntas
├── qa_utils.py                  # Utilidades
├── qa_dataset_generator.py      # Gerador de datasets
├── qa_inference.py              # Engine de inferência
├── qa_evaluator.py              # Avaliador
├── qa_pipeline.py               # Pipeline principal
├── __init__.py                  # Imports
├── example_qa_usage.py          # Exemplos
├── requirements_qa.txt          # Dependências
├── FINE_TUNING_GUIDE.md         # Guia de fine-tuning
├── README.md                    # Este ficheiro
├── output/                      # Outputs gerados
│   ├── qa_results.json
│   ├── qa_dataset.json
│   └── qa_dataset_squad.json
└── models/                      # Modelos fine-tuned (opcional)
    └── bertimbau-finetuned/
```

---

## Integração no Pipeline Principal

Para integrar QA no pipeline NLP completo:

```python
# pipeline.py (do projeto principal)
from qa import QAPipeline

class EmailNLPPipeline:
    def __init__(self):
        self.qa_pipeline = QAPipeline()
    
    def process(self, email):
        # 1. Preprocessing
        processed_email = self.preprocess(email)
        
        # 2. Intent classification
        intent = self.classify_intent(processed_email)
        
        # 3. Trigger extraction
        triggers = self.extract_triggers(processed_email)
        
        # 4. QA para informações estruturadas
        qa_result = self.qa_pipeline.process_email(processed_email)
        
        # 5. Temporal normalization
        temporal = self.normalize_temporal(processed_email)
        
        return {
            'intent': intent,
            'triggers': triggers,
            'meeting_info': qa_result.get_answers_only(),
            'temporal': temporal
        }
```

---

## Contribuições & Melhorias

Sugestões de melhorias:

- [ ] Suporte para XLM-RoBERTa (multilíngue melhorado)
- [ ] Confidence calibration com temperatura
- [ ] Few-shot learning para novas categorias
- [ ] Ensemble de múltiplos modelos
- [ ] API REST para inferência
- [ ] Interface web
- [ ] Logging avançado

---

## Licença

Projeto académico. Desenvolvido para Dissertação de Mestrado.

---

## Contacto

**Projeto**: Email Recognition PT-PT
**Módulo**: Question Answering
**Versão**: 1.0
**Data**: Maio 2026

---

## Referências

- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [BERTimbau](https://arxiv.org/abs/2009.06241)
- [Question Answering with HuggingFace](https://huggingface.co/course/chapter7/2)
- [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)
- [Portuguese NLP Resources](https://github.com/davidsbatista/NLP-for-Portuguese)
