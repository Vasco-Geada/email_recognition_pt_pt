"""
INTEGRATION_GUIDE.md

Guia de Integração do Módulo QA com o Pipeline NLP

Este documento mostra como integrar o módulo de Question Answering
com o pipeline de NLP existente do projeto.

Project: Email Recognition PT-PT
Version: 1.0
"""

# Integração do Módulo QA

## Índice

1. [Visão Geral](#visão-geral)
2. [Arquitetura do Pipeline](#arquitetura-do-pipeline)
3. [Integração Passo-a-Passo](#integração-passo-a-passo)
4. [Exemplos de Código](#exemplos-de-código)
5. [Fluxo de Dados End-to-End](#fluxo-de-dados-end-to-end)
6. [Handling de Erros](#handling-de-erros)
7. [Performance](#performance)

---

## Visão Geral

### Componentes do Pipeline NLP Atual

```
Email
  ↓
[Preprocessing] ← email_pipeline.py
  ├─ Limpeza de texto
  ├─ Normalização
  └─ Tokenização
  ↓
[Intent Classification] ← intent_classifier.py, train_intent.py
  ├─ agendamento_reuniao
  ├─ cancelamento_reuniao
  └─ ...
  ↓
[Trigger Extraction] ← trigger_extraction.py
  ├─ reunir, marcar, etc
  ↓
[Argument Extraction] ← argument_extraction.py
  ├─ participants: [] (NER)
  ├─ time_expressions: [] (regex)
  └─ locations: [] (regex)
  ↓
[Temporal Normalization] ← temporal_normalization.py
  ├─ Normalizar expressões temporais
  └─ Converter para formato estruturado
  ↓
Output: Structured Meeting Information
```

### Onde o QA se encaixa

O QA complementa a extração existente:

```
Email
  ↓
[Preprocessing]
  ↓
[Intent + Trigger + Arguments] ← Pipeline Existente
  ↓
[QA Module] ← NOVO
  ├─ Responder perguntas estruturadas
  ├─ Validar/complementar respostas anteriores
  └─ Fornecer alternativa a regex
  ↓
[Temporal Normalization]
  ↓
[Fusion/Consensus] ← NOVO
  ├─ Combinar resultados
  └─ Resolver conflitos
  ↓
Output: Final Structured Information
```

---

## Arquitetura do Pipeline

### Versão Integrada Proposta

```python
class EmailNLPPipeline:
    """Pipeline NLP integrado com QA."""
    
    def __init__(self):
        # Componentes existentes
        self.preprocessor = EmailPreprocessor()
        self.intent_classifier = IntentClassifier()
        self.trigger_extractor = TriggerExtractor()
        self.arg_extractor = ArgumentExtractor()
        self.temporal_normalizer = TemporalNormalizer()
        
        # Componente novo: QA
        self.qa_pipeline = QAPipeline()  # ← NOVO
        
        # Fusão de resultados
        self.result_fusion = ResultFusion()  # ← NOVO
    
    def process(self, email: Dict) -> ProcessedEmail:
        """Processa email através de todo o pipeline."""
        
        # 1. Preprocessing (existente)
        preprocessed = self.preprocessor.process(email)
        
        # 2. Intent + Trigger (existente)
        intent = self.intent_classifier.classify(preprocessed)
        triggers = self.trigger_extractor.extract(preprocessed)
        
        # 3. Argument Extraction (existente)
        arguments = self.arg_extractor.extract(preprocessed)
        
        # 4. QA (NOVO)
        qa_result = self.qa_pipeline.process_email(
            email_text=email['text'],
            email_id=email.get('id'),
        )
        
        # 5. Temporal Normalization (existente)
        temporal = self.temporal_normalizer.normalize(arguments['time'])
        
        # 6. Fusão de resultados (NOVO)
        final_result = self.result_fusion.fuse(
            intent=intent,
            triggers=triggers,
            arguments=arguments,
            qa_result=qa_result,
            temporal=temporal,
        )
        
        return final_result
```

---

## Integração Passo-a-Passo

### Passo 1: Instalar Dependências QA

```bash
# Adicionar ao requirements.txt principal
pip install -r qa/requirements_qa.txt

# Ou manualmente
pip install torch transformers
```

### Passo 2: Importar Módulo QA

```python
# No ficheiro principal do pipeline
from qa.qa_pipeline import QAPipeline, EmailQAResult
from qa.qa_evaluator import QAEvaluator

# Ou usar imports específicos
from qa import (
    QAPipeline,
    QAQuestions,
    QuestionCategory,
)
```

### Passo 3: Inicializar Pipeline QA

```python
class EmailNLPPipeline:
    def __init__(self, use_qa: bool = True):
        # ... componentes existentes ...
        
        if use_qa:
            self.qa_pipeline = QAPipeline(
                model_name='bertimbau-pt',
                device='cuda' if torch.cuda.is_available() else 'cpu',
                confidence_threshold=0.5,
            )
        else:
            self.qa_pipeline = None
```

### Passo 4: Integrar no Fluxo

```python
def process_email(self, email: Dict) -> Dict:
    """Processa email com QA integrado."""
    
    # 1. Preprocessing
    text = self.preprocess(email['text'])
    
    # 2-3. Intent, Trigger, Arguments (existente)
    intent = self.classify_intent(text)
    triggers = self.extract_triggers(text)
    arguments = self.extract_arguments(text)
    
    # 4. QA (NOVO)
    qa_result = None
    if self.qa_pipeline and intent == 'agendamento_reuniao':
        qa_result = self.qa_pipeline.process_email(
            email_text=email['text'],
            email_id=email.get('id'),
        )
    
    # 5. Temporal normalization
    temporal = self.normalize_temporal(arguments['time'])
    
    # 6. Combinar resultados
    final = self._fuse_results(
        intent=intent,
        arguments=arguments,
        qa_result=qa_result,
        temporal=temporal,
    )
    
    return final

def _fuse_results(self, intent, arguments, qa_result, temporal):
    """Funde resultados de múltiplas fontes."""
    
    result = {
        'intent': intent,
        'participants': arguments['participants'],
        'time': arguments['time'],
        'location': arguments['location'],
        'topic': arguments['topic'],
        'temporal_normalized': temporal,
    }
    
    # Se QA disponível, usar para validar/complementar
    if qa_result:
        qa_answers = qa_result.get_answers_only()
        
        # Validar participants
        if not result['participants'] and qa_answers.get('participants'):
            result['participants'] = qa_answers['participants']
            result['participants_source'] = 'qa'
        
        # Complementar time
        if not result['time'] and qa_answers.get('time'):
            result['time'] = qa_answers['time']
            result['time_source'] = 'qa'
        
        # Usar location de QA se não houver
        if not result['location'] and qa_answers.get('location'):
            result['location'] = qa_answers['location']
            result['location_source'] = 'qa'
        
        # Adicionar topic (nunca extraído por regex)
        if qa_answers.get('topic'):
            result['topic'] = qa_answers['topic']
            result['topic_source'] = 'qa'
        
        result['qa_confidences'] = {
            cat: qa_result.qa_results[cat].get('confidence', 0.0)
            for cat in ['participants', 'time', 'location', 'topic']
        }
    
    return result
```

### Passo 5: Atualizar Fluxo de Batch

```python
def process_batch(self, emails: List[Dict]) -> List[Dict]:
    """Processa batch com suporte a QA."""
    
    results = []
    
    for i, email in enumerate(emails):
        if (i + 1) % 10 == 0:
            print(f"Processados {i + 1}/{len(emails)}")
        
        result = self.process_email(email)
        results.append(result)
    
    return results
```

---

## Exemplos de Código

### Exemplo 1: Integração Simples

```python
from preprocessing.email_pipeline_enhanced import EmailPipeline
from qa.qa_pipeline import QAPipeline
import json

# Carregar gold annotations
with open('gold_annotations/output/gold.json') as f:
    emails = json.load(f)

# Inicializar pipelines
nlp_pipeline = EmailPipeline()
qa_pipeline = QAPipeline(device='cpu')

# Processar
for email in emails[:5]:
    # Fluxo NLP original
    nlp_result = nlp_pipeline.process(email['text'])
    
    # Adicionar QA
    qa_result = qa_pipeline.process_email(
        email_text=email['text'],
        email_id=email['id'],
    )
    
    # Combinar
    combined = {
        **nlp_result,
        'qa_answers': qa_result.get_answers_only() if qa_result else {},
    }
    
    print(f"\nEmail {email['id']}:")
    print(f"  Intent: {nlp_result['intent']}")
    print(f"  Participants (NLP): {nlp_result['participants']}")
    print(f"  Participants (QA): {combined['qa_answers'].get('participants')}")
```

### Exemplo 2: Comparação NLP vs QA

```python
from qa.qa_evaluator import QAEvaluator, ComparativeAnalyzer
from qa.qa_utils import MetricsCalculator

# Simular resultados NLP (baseline)
nlp_results = [
    {
        'id': '1',
        'predicted': 'Ana',
        'reference': 'Ana',
        'category': 'participants',
    },
    {
        'id': '2',
        'predicted': '',  # NLP não conseguiu extrair
        'reference': 'sexta às 15h',
        'category': 'time',
    },
]

# QA results
qa_results = [
    {
        'id': '1',
        'predicted': 'Ana',
        'reference': 'Ana',
        'category': 'participants',
    },
    {
        'id': '2',
        'predicted': 'sexta às 15h',  # QA conseguiu
        'reference': 'sexta às 15h',
        'category': 'time',
    },
]

# Avaliar
evaluator_nlp = QAEvaluator()
evaluator_qa = QAEvaluator()

for item in nlp_results:
    evaluator_nlp.evaluate_example(
        example_id=item['id'],
        question='',
        predicted=item['predicted'],
        reference=item['reference'],
        category=item['category'],
    )

for item in qa_results:
    evaluator_qa.evaluate_example(
        example_id=item['id'],
        question='',
        predicted=item['predicted'],
        reference=item['reference'],
        category=item['category'],
    )

# Comparar
print("\n=== NLP Baseline ===")
evaluator_nlp.print_report()

print("\n=== QA Results ===")
evaluator_qa.print_report()

comparison = ComparativeAnalyzer.compare_methods({
    'nlp': evaluator_nlp.metrics_list,
    'qa': evaluator_qa.metrics_list,
})

print("\n=== Comparison ===")
for method, metrics in comparison.items():
    print(f"{method}:")
    print(f"  EM: {metrics['em']:.3f}")
    print(f"  F1: {metrics['f1']:.3f}")
```

### Exemplo 3: Evaluação com Gold Annotations

```python
from qa import QAPipeline, QAEvaluator
import json

# Carregar gold
with open('gold_annotations/output/gold.json') as f:
    gold = json.load(f)

# Processar com QA
pipeline = QAPipeline()
qa_predictions = []

for annotation in gold:
    result = pipeline.process_email(
        email_text=annotation['text'],
        email_id=annotation['id'],
    )
    
    if result:
        answers = result.get_answers_only()
        for category, answer in answers.items():
            qa_predictions.append({
                'id': f"{annotation['id']}_{category}",
                'category': category,
                'predicted': answer,
                'confidence': result.qa_results[category]['confidence'],
            })

# Avaliar contra gold annotations
evaluator = QAEvaluator()

for pred in qa_predictions:
    category = pred['category']
    
    # Encontrar referência
    email_id = int(pred['id'].split('_')[0])
    gold_annotation = next(g for g in gold if g['id'] == email_id)
    gold_answers = gold_annotation['arguments'][category]
    
    reference = gold_answers[0] if gold_answers else ''
    
    evaluator.evaluate_example(
        example_id=pred['id'],
        question='',
        predicted=pred['predicted'] or '',
        reference=reference,
        confidence=pred['confidence'],
        category=category,
    )

evaluator.print_report()
evaluator.save_results('qa/output/evaluation_vs_gold.json')
```

---

## Fluxo de Dados End-to-End

### Cenário: Email sobre Reunião

```
Input Email:
┌─────────────────────────────────────────────────────┐
│ Boas Ana,                                           │
│ Consegues reunir sexta às 15h no Teams para        │
│ discutir o dataset?                                │
│ Obrigado                                            │
└─────────────────────────────────────────────────────┘

├─ ID: 1
├─ Subject: "Re: Dissertação"
└─ Pessoa: "aluno"

                         │
                         ▼
             [Preprocessing]
        ├─ Remove stopwords
        ├─ Lowercase
        └─ Tokenize
                         │
                         ▼
        [Intent Classification]
        └─ Resultado: "agendamento_reuniao"
                         │
                         ▼
        [Trigger Extraction]
        └─ Triggers: ["reunir"]
                         │
                         ▼
        [Argument Extraction - NLP Existente]
        ├─ Participants: [] (não encontrou NER)
        ├─ Time: [] (regex não matched)
        ├─ Location: [] (não encontrou)
        └─ Topic: [] (não encontrou)
                         │
                         ▼
        [QA Pipeline - NOVO]
        
        Query 1: "Quem participa na reunião?"
        ├─ BERT QA
        ├─ Answer: "Ana"
        └─ Confidence: 0.92
        
        Query 2: "Quando é a reunião?"
        ├─ BERT QA
        ├─ Answer: "sexta às 15h"
        └─ Confidence: 0.88
        
        Query 3: "Onde é a reunião?"
        ├─ BERT QA
        ├─ Answer: "Teams"
        └─ Confidence: 0.85
        
        Query 4: "Qual é o tópico?"
        ├─ BERT QA
        ├─ Answer: "dataset"
        └─ Confidence: 0.80
                         │
                         ▼
        [Temporal Normalization]
        ├─ Input: "sexta às 15h"
        ├─ Parse: {day: "friday", time: "15:00"}
        └─ Normalize: "2026-05-16T15:00:00"
                         │
                         ▼
        [Result Fusion]
        ├─ Participants: "Ana" (from QA)
        ├─ Time: "sexta às 15h" (from QA)
        ├─ Location: "Teams" (from QA)
        ├─ Topic: "dataset" (from QA)
        ├─ Temporal: "2026-05-16T15:00:00"
        └─ Confidence scores: {0.92, 0.88, 0.85, 0.80}
                         │
                         ▼
Output (Structured):
┌──────────────────────────────────────────────┐
│ {                                            │
│   "id": 1,                                   │
│   "intent": "agendamento_reuniao",           │
│   "participants": "Ana",                     │
│   "time": "sexta às 15h",                    │
│   "time_normalized": "2026-05-16T15:00:00",  │
│   "location": "Teams",                       │
│   "topic": "dataset",                        │
│   "sources": {                               │
│     "participants": "qa",                    │
│     "time": "qa",                            │
│     "location": "qa",                        │
│     "topic": "qa"                            │
│   },                                         │
│   "confidences": {                           │
│     "participants": 0.92,                    │
│     "time": 0.88,                            │
│     "location": 0.85,                        │
│     "topic": 0.80                            │
│   }                                          │
│ }                                            │
└──────────────────────────────────────────────┘
```

---

## Handling de Erros

### Tratamento Integrado

```python
class RobustEmailPipeline:
    """Pipeline com tratamento robusto de erros."""
    
    def process_email(self, email: Dict) -> Dict:
        """Processa email com fallback."""
        
        try:
            # 1. Preprocessing
            text = self.preprocess(email['text'])
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return {'error': 'preprocessing_failed'}
        
        try:
            # 2. NLP components
            intent = self.classify_intent(text)
            arguments = self.extract_arguments(text)
        except Exception as e:
            logger.error(f"NLP failed: {e}")
            # Continua com QA mesmo se NLP falhar
            arguments = {'participants': [], 'time': [], 'location': [], 'topic': []}
        
        # 3. QA (com fallback)
        qa_result = None
        try:
            if self.qa_pipeline:
                qa_result = self.qa_pipeline.process_email(email['text'])
        except torch.cuda.OutOfMemoryError:
            logger.warning("GPU OOM, switching to CPU for QA")
            self.qa_pipeline.qa_engine.device = 'cpu'
            qa_result = self.qa_pipeline.process_email(email['text'])
        except Exception as e:
            logger.error(f"QA failed: {e}")
            qa_result = None  # Continua sem QA
        
        # 4. Fuse results
        final = self._fuse_results(intent, arguments, qa_result)
        
        return final
```

### Configuração de Fallback

```python
class QAPipelineWithFallback:
    """QA pipeline com múltiplos fallbacks."""
    
    def __init__(self):
        try:
            self.primary = QAPipeline(model_name='bertimbau-pt')
            self.available = True
        except Exception:
            logger.warning("Primary model failed, using multilingual fallback")
            try:
                self.primary = QAPipeline(model_name='multilingual', device='cpu')
                self.available = True
            except Exception:
                logger.error("All QA models failed")
                self.available = False
    
    def process_email(self, email_text: str):
        if not self.available:
            return None
        return self.primary.process_email(email_text)
```

---

## Performance

### Benchmark de Performance

```python
import time
import statistics

# Pipeline integrado
pipeline = EmailNLPPipeline(use_qa=True)

emails = load_emails(100)
times = []

for email in emails:
    start = time.time()
    result = pipeline.process_email(email)
    elapsed = time.time() - start
    times.append(elapsed)

print(f"Average: {statistics.mean(times):.3f}s")
print(f"Median: {statistics.median(times):.3f}s")
print(f"Stdev: {statistics.stdev(times):.3f}s")
print(f"Min: {min(times):.3f}s")
print(f"Max: {max(times):.3f}s")
```

### Otimizações Recomendadas

1. **GPU**: Usar GPU reduz de ~1s para ~0.1s por email
2. **Caching**: Resultados em cache reduzem reprocessamento
3. **Batch**: Processar em batches é mais eficiente
4. **Lazy Loading**: Carregar modelos sob demanda

---

## Resumo de Integração

| Componente | Como Integrar | Tempo |
|-----------|---------------|-------|
| **1. Imports** | Add `from qa import ...` | 5 min |
| **2. Dependencies** | `pip install -r qa/requirements_qa.txt` | 5 min |
| **3. Initialize** | Create `QAPipeline()` instance | 5 min |
| **4. Process** | Call `qa_pipeline.process_email()` | 30 min |
| **5. Fuse Results** | Implement `_fuse_results()` | 30 min |
| **6. Test** | Test end-to-end | 30 min |
| **7. Evaluate** | Compare com baseline | 30 min |
| **TOTAL** | | ~2-3 horas |

---

## Próximos Passos

1. ✅ Instalar módulo QA
2. ✅ Integrar no pipeline
3. ✅ Fazer processamento básico
4. ⭐ Avaliar performance (vs baseline NLP)
5. ⭐ Fine-tune modelo com dados do projeto
6. ⭐ Experimentar técnicas de ensemble
7. ⭐ Otimizar latência (batch, GPU, cache)
8. ⭐ Deploy em produção

---

**Documentação Completa**: Ver [README.md](README.md) e [ARCHITECTURE.md](ARCHITECTURE.md)

**Exemplos**: Ver [example_qa_usage.py](example_qa_usage.py)

**Fine-Tuning**: Ver [FINE_TUNING_GUIDE.md](FINE_TUNING_GUIDE.md)
