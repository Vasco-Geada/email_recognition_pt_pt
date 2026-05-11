# ÍNDICE - Sistema de Gold Annotations

Documentação e estrutura completa do sistema de geração de gold annotations.

## 📋 Ficheiros Principais

### Módulos Core

| Ficheiro | Descrição | Linhas | Responsabilidades |
|----------|-----------|--------|------------------|
| `heuristic_extractors.py` | Extractores heurísticos | ~450 | TriggerExtractor, ParticipantExtractor, TemporalExtractor, LocationExtractor, TopicExtractor, HeuristicAnnotationExtractor |
| `validators.py` | Validadores e normalizadores | ~400 | AnnotationValidator, JSONValidator, ConsistencyValidator |
| `gold_annotations_generator.py` | Orquestrador principal | ~380 | GoldAnnotationsGenerator (pipeline completo) |
| `evaluate_annotations.py` | Avaliador com métricas | ~420 | AnnotationEvaluator, ClassificationMetrics, EvaluationResult |
| `config.py` | Configurações globais | ~220 | Patterns, triggers, locations, intents, confidence |
| `__init__.py` | Package exports | ~40 | Imports públicos |

### Ficheiros de Teste e Exemplo

| Ficheiro | Descrição | Responsabilidades |
|----------|-----------|------------------|
| `test_gold_annotations.py` | Testes unitários | TestTriggerExtractor, TestParticipantExtractor, TestValidator, TestGenerator, TestEvaluator |
| `example_usage.py` | Demonstração completa | 5 exemplos funcionais + sample dataset |
| `setup.py` | Setup e verificação | Verificação de instalação, setup inicial |

### Documentação

| Ficheiro | Tipo | Público | Conteúdo |
|----------|------|---------|----------|
| `README.md` | Markdown | Sim | Visão geral, quick start, API, exemplos |
| `QUICKSTART.md` | Markdown | Sim | 5 minutos de inicio (ultra rápido) |
| `TECHNICAL_DOCUMENTATION.md` | Markdown | Sim | Arquitetura, algoritmos, design patterns |
| `ÍNDICE.md` | Markdown | Sim | Este ficheiro - mapa completo |
| `requirements.txt` | Text | Sim | Dependências (nenhuma obrigatória) |

## 🏗️ Arquitetura

### Camadas do Sistema

```
┌─────────────────────────────────────┐
│   Interface do Utilizador           │
│  (CLI / Python API / Example Scripts)│
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   GoldAnnotationsGenerator          │
│   (Orquestrador / Pipeline)         │
└──────────────┬──────────────────────┘
               │
        ┌──────┴──────┐
        │             │
┌───────▼────────┐ ┌─▼──────────────┐
│  Extractores   │ │  Validadores   │
│  Heurísticos   │ │  & Normalizador│
├────────────────┤ ├────────────────┤
│ - Trigger      │ │ - Annotation   │
│ - Participant  │ │ - JSON         │
│ - Temporal     │ │ - Consistency  │
│ - Location     │ │                │
│ - Topic        │ │                │
└────────────────┘ └────────────────┘
        │                 │
        └────────┬────────┘
                 │
        ┌────────▼───────────┐
        │  Config (Patterns) │
        │  & Constantes      │
        └────────────────────┘
               │
        ┌──────┴──────────┐
        │                 │
   ┌────▼──────┐    ┌─────▼────┐
   │  Output   │    │  Evaluator│
   │  (JSON)   │    │ (Metrics) │
   └───────────┘    └───────────┘
```

### Fluxo de Dados

```
Input JSON
   ↓ (emails: subject, body, label)
GoldAnnotationsGenerator.run()
   ├─ load_emails_json()
   ├─ process_batch()
   │   ├─ TriggerExtractor.extract()
   │   ├─ ParticipantExtractor.extract()
   │   ├─ TemporalExtractor.extract()
   │   ├─ LocationExtractor.extract()
   │   └─ TopicExtractor.extract()
   │       → ExtractionResult(values, confidence, metadata)
   ├─ validate_annotations()
   │   ├─ AnnotationValidator.validate_batch()
   │   ├─ AnnotationValidator.normalize_annotation()
   │   └─ ConsistencyValidator.check_intent_trigger_consistency()
   ├─ save_annotations()
   └─ generate_report()
       → Output JSON (anotações estruturadas)
              ↓
AnnotationEvaluator (com predições)
   ├─ evaluate_batch()
   ├─ ClassificationMetrics.precision
   ├─ ClassificationMetrics.recall
   ├─ ClassificationMetrics.f1
   ├─ confusion_matrix
   └─ error_analysis
       → Evaluation Report (JSON + tabelas)
```

## 📊 Estrutura de Dados

### Input JSON (Emails)

```json
[
  {
    "subject": "string",
    "body": "string",
    "label": "agendamento_reuniao | cancelamento_reuniao | reuniao_confirmada"
  }
]
```

### Output JSON (Gold Annotations)

```json
[
  {
    "id": 1,
    "text": "string",
    "intent": "string",
    "trigger": ["string"],
    "arguments": {
      "participants": ["string"],
      "time": ["string"],
      "location": ["string"],
      "topic": ["string"]
    },
    "confidence": {
      "trigger": 0.0-1.0,
      "participants": 0.0-1.0,
      "temporal": 0.0-1.0,
      "location": 0.0-1.0,
      "topic": 0.0-1.0
    },
    "metadata": {
      "source_subject": "string",
      "extracted_at": "ISO-8601",
      "extraction_method": "heuristic",
      "reviewed": false,
      "reviewer": null,
      "version": "1.0"
    }
  }
]
```

### Evaluation Report JSON

```json
{
  "summary": {
    "total_samples": 0,
    "exact_matches": 0,
    "exact_match_accuracy": 0.0
  },
  "intent_metrics": {
    "agendamento_reuniao": {
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "tp": 0,
      "fp": 0,
      "fn": 0
    }
  },
  "argument_metrics": {...},
  "overall_metrics": {...},
  "confusion_matrix": {...},
  "error_analysis": {...}
}
```

## 🔧 Classes Principais

### HeuristicAnnotationExtractor

```python
class HeuristicAnnotationExtractor:
    trigger_extractor: TriggerExtractor
    participant_extractor: ParticipantExtractor
    temporal_extractor: TemporalExtractor
    location_extractor: LocationExtractor
    topic_extractor: TopicExtractor
    
    def extract_all(text, subject) -> Dict[str, ExtractionResult]
```

### AnnotationValidator

```python
class AnnotationValidator:
    REQUIRED_FIELDS = {}
    VALID_INTENTS = []
    ARGUMENT_FIELDS = []
    
    def validate_single_annotation(annotation) -> ValidationResult
    def validate_batch(annotations) -> ValidationResult
    def normalize_annotation(annotation) -> Dict
```

### GoldAnnotationsGenerator

```python
class GoldAnnotationsGenerator:
    extractor: HeuristicAnnotationExtractor
    validator: AnnotationValidator
    
    def load_emails_json(filepath) -> List[Dict]
    def process_batch(emails) -> Tuple[List[Dict], List[str]]
    def validate_annotations(annotations) -> Tuple[List[Dict], List[str]]
    def save_annotations(annotations, output_path) -> bool
    def generate_report(annotations, errors) -> Dict
    def run(input_json, output_json) -> bool
```

### AnnotationEvaluator

```python
class AnnotationEvaluator:
    intent_metrics: Dict[str, ClassificationMetrics]
    argument_metrics: Dict[str, ClassificationMetrics]
    
    def evaluate_batch(gold, predictions) -> EvaluationResult
    def print_metrics(result) -> None
    def save_report(result, output_path) -> bool
```

## 📈 Métricas Implementadas

### Classification Metrics

- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall)
- **Accuracy**: (TP + TN) / Total

### Aggregation Levels

1. **Per Intent**: Métricas para cada classe (agendamento, cancelamento, confirmação)
2. **Per Argument**: Métricas para cada tipo de argumento (participants, time, location, topic)
3. **Overall**: Agregação global

### Error Analysis

- Confusion matrix (intent predictions vs gold)
- False positives, false negatives analysis
- Missing vs extra predictions
- Detailed error examples

## 🚀 APIs e Interface

### Command Line Interface (CLI)

```bash
# Gerar gold annotations
python gold_annotations_generator.py input.json output.json [-v | -q]

# Avaliar anotações
python evaluate_annotations.py gold.json predictions.json [-o report.json] [-v]

# Setup e verificação
python setup.py [verify|install|test|demo|usage]

# Executar testes
python test_gold_annotations.py

# Executar exemplos
python example_usage.py
```

### Python API

```python
# Importar módulos
from gold_annotations import (
    GoldAnnotationsGenerator,
    AnnotationValidator,
    AnnotationEvaluator,
    HeuristicAnnotationExtractor,
)

# Gerar annotations
generator = GoldAnnotationsGenerator(verbose=True)
success = generator.run('input.json', 'output.json')

# Validar
validator = AnnotationValidator()
result = validator.validate_batch(annotations)

# Avaliar
evaluator = AnnotationEvaluator()
eval_result = evaluator.evaluate_batch(gold, predictions)
evaluator.print_metrics(eval_result)
```

## 🔍 Algoritmos Implementados

### Trigger Detection

1. **Regex Pattern Matching** (Default)
   - Triggers: reunir, marcar, cancelar, confirmar, agendar, combinar, encontrar, falar, discutir
   - Confidence: 0.90

### Participant Detection

1. **Salutation-based**
   - Padrões: "Olá Ana", "Professor Silva"
   - Confidence: 0.75

2. **Proper Noun Extraction**
   - Regras: capitalized words em contexto
   - Filtros: nomes muito curtos, títulos

### Temporal Expression Detection

1. **Relative Days**
   - amanhã, hoje, ontem, sexta, segunda, etc.
   
2. **Time Expressions**
   - 15h, às 15:00, meio-dia
   
3. **Time Periods**
   - manhã, tarde, noite, depois de almoço

### Location Detection

1. **Online Platforms**
   - Teams, Zoom, Discord, Skype, Google Meet
   
2. **Physical Spaces**
   - Salas, auditórios, biblioteca, laboratório

### Topic Detection

1. **Academic Terms**
   - dissertação, tese, relatório, pipeline, NLP, BERT, F1, etc.

## 📚 Documentação por Tópico

| Tópico | Ficheiro | Secção |
|--------|----------|--------|
| Visão Geral | README.md | Tudo |
| Quick Start (5 min) | QUICKSTART.md | Tudo |
| Arquitetura | TECHNICAL_DOCUMENTATION.md | Architecture |
| Algoritmos | TECHNICAL_DOCUMENTATION.md | Extraction Algorithms |
| Design Patterns | TECHNICAL_DOCUMENTATION.md | Design Patterns |
| Performance | TECHNICAL_DOCUMENTATION.md | Performance |
| Troubleshooting | TECHNICAL_DOCUMENTATION.md | Troubleshooting |
| API Python | README.md | Advanced Config |
| CLI | QUICKSTART.md | 5 Minutos |
| Exemplos | README.md | Examples |
| Testes | Inline code | test_gold_annotations.py |

## ✅ Checklist de Funcionalidades

### Extração Implementada
- [x] Trigger extraction (9 triggers)
- [x] Participant extraction (names + titles)
- [x] Temporal expression extraction (5 types)
- [x] Location extraction (4 categories)
- [x] Topic extraction (3 categories)

### Validação Implementada
- [x] Field validation
- [x] Type checking
- [x] Intent validation
- [x] UTF-8 encoding check
- [x] Duplicate detection
- [x] Consistency checking
- [x] Normalization

### Avaliação Implementada
- [x] Exact Match
- [x] Precision per intent
- [x] Recall per intent
- [x] F1-Score per intent
- [x] Confusion matrix
- [x] Error analysis
- [x] Argument-level metrics
- [x] Overall metrics

### Recursos Opcionais (Bónus)
- [x] Confidence scoring
- [x] Metadata tracking
- [x] Report generation
- [x] CLI interface
- [x] Python API
- [x] Setup verification
- [x] Unit tests
- [x] Example scripts

### Não Implementados (Future Work)
- [ ] BIO tagging export
- [ ] spaCy DocBin format
- [ ] HuggingFace datasets integration
- [ ] Interactive revision UI
- [ ] Machine learning-based extraction
- [ ] Multi-language support (beyond PT-PT)

## 📊 Estatísticas do Projeto

```
Total de Linhas de Código: ~2500+
Ficheiros Python: 8
Ficheiros de Documentação: 5
Módulos Importados: stdlib only
Comentários: ~400+ linhas
Docstrings: 100% coverage
Testes: 20+ casos
Exemplos: 5 demonstrações completas
```

## 🎯 Próximos Passos (Futuro)

1. **Melhorias de Extraction**
   - Usar spaCy para NER
   - Temporal normalization com TIMEX
   - SRL para argumentos

2. **Otimizações**
   - Cache de padrões compilados
   - Processamento paralelo

3. **Integração**
   - Export para spaCy
   - Export para HF datasets
   - WebUI para revisão manual

4. **Pesquisa**
   - Comparar com baselines
   - Publicar metodologia
   - Dados públicos

## 📝 Citação

Se usar este sistema em pesquisa académica, cite como:

```bibtex
@software{email_recognition_2024,
  title={Gold Annotations System for Email Recognition in Portuguese},
  author={Email Recognition PT-PT Project},
  year={2024},
  url={https://github.com/projeto/email_recognition_pt_pt}
}
```

---

**Versão:** 1.0  
**Data:** 11 de Maio de 2024  
**Linguagem:** Python 3.11+  
**Licença:** Open Source
