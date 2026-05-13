# Documentação Técnica - Sistema de Gold Annotations

## Arquitetura

### Componentes Principais

```
gold_annotations/
├── heuristic_extractors.py      # Extractores (TriggerExtractor, ParticipantExtractor, etc.)
├── validators.py                 # Validação (AnnotationValidator, JSONValidator, etc.)
├── gold_annotations_generator.py # Orquestrador principal
├── evaluate_annotations.py       # Avaliador com métricas científicas
├── __init__.py                   # Exports principais
├── example_usage.py              # Demonstração completa
├── test_gold_annotations.py      # Testes unitários
└── README.md, QUICKSTART.md      # Documentação
```

### Fluxo de Dados

```
Raw Emails JSON
       ↓
  Carregamento
       ↓
  Extração Heurística
       ├── TriggerExtractor
       ├── ParticipantExtractor
       ├── TemporalExtractor
       ├── LocationExtractor
       └── TopicExtractor
       ↓
  Validação & Normalização
       ├── AnnotationValidator
       ├── ConsistencyValidator
       └── JSONValidator
       ↓
  Gold Annotations JSON (com confidence scores)
       ↓
  Revisão Manual (opcional)
       ↓
  Exportação para Avaliação
       ↓
  Avaliação com Predições de Modelo
       └── AnnotationEvaluator
           ├── Precision/Recall/F1
           ├── Confusion Matrix
           └── Error Analysis
```

## Design Patterns Utilizados

### 1. **Extractor Pattern**

Cada tipo de informação tem seu extractor isolado:

```python
class TriggerExtractor:
    TRIGGER_PATTERNS = {...}
    def extract(self, text: str) -> ExtractionResult
```

**Benefícios:**
- Separação de responsabilidades
- Fácil de estender (adicionar novos triggers)
- Testável independentemente

### 2. **Result Wrapper Pattern**

Resultados de extração incluem confiança e metadados:

```python
@dataclass
class ExtractionResult:
    values: List[str]           # Valores extraídos
    confidence: float            # Confiança (0.0 a 1.0)
    metadata: Dict = field(...)  # Contexto adicional
```

### 3. **Validator Chain Pattern**

Múltiplos validadores podem ser aplicados:

```python
class AnnotationValidator:
    - validate_single_annotation()
    - validate_batch()
    - normalize_annotation()
```

### 4. **Metrics Aggregator Pattern**

Métricas são agregadas e calculadas em tempo real:

```python
@dataclass
class ClassificationMetrics:
    @property
    def precision(self): ...
    @property
    def recall(self): ...
    @property
    def f1(self): ...
```

## Algoritmos de Extração

### TriggerExtractor

**Abordagem:** Regex-based pattern matching

**Padrões:**
```python
{
    'reunir': [r'\breunir(?:mos|em|ão|ia)?', ...],
    'marcar': [r'\bmarcar(?:\s+(?:uma|a|o))?', ...],
    'cancelar': [r'\bcancelar', r'\bcancelamento', ...],
    ...
}
```

**Características:**
- Case-insensitive
- Suporta variações morfológicas
- Retorna primeiro trigger encontrado
- Confidence: 0.9

### ParticipantExtractor

**Abordagem:** Salutation detection + proper noun extraction

**Padrões:**
```python
[
    r'(?:Olá|Oi|Caro|...)\s+([A-Z][a-zá-ú]+(?:\s+[A-Z][a-zá-ú]+)?)',  # Salutations
    r'(?:com|a|para|...)\s+(?:Professor|Dr\.|...)\s+([A-Z][a-zá-ú]+)',  # Titles
    r'@([a-zA-Z][a-zA-Z0-9]*)',  # Mentions
]
```

**Filtros:**
- Comprimento mínimo > 1
- Remove títulos isolados
- Deduplica resultados

### TemporalExtractor

**Abordagem:** Padrão categórico por tipo temporal

**Categorias:**
1. Dias relativos (amanhã, sexta, etc.)
2. Semanas (próxima semana, semana que vem)
3. Horas (15h, às 15:00, meio-dia)
4. Períodos (manhã, tarde, depois de almoço)
5. Expressões relativas (mais logo, em breve)

**Regex Examples:**
```python
r'\b(?:amanhã|hoje|ontem)\b'              # Dias
r'\b(?:segunda|terça|quarta|...)(?: -feira)?\b'  # Dias semana
r'(?:às|ao|a)\s*(\d{1,2})(?:[:h])?\s*(\d{2})?\s*h'  # Horas
r'\b(?:de\s+)?(?:manhã|tarde|noite)\b'   # Períodos
```

### LocationExtractor

**Abordagem:** Categorização por tipo

**Categorias:**
1. Plataformas (Teams, Zoom, Discord)
2. Salas (sala 2.3, auditório)
3. Espaços (biblioteca, laboratório)
4. Edifícios (bloco A, departamento)

### TopicExtractor

**Abordagem:** Lexicon-based matching

**Categorias:**
1. Trabalhos académicos (dissertação, tese, relatório)
2. Conceitos técnicos (BERT, NLP, pipeline, F1)
3. Disciplinas (linguística, IA)

## Métricas de Avaliação

### Intent Evaluation

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * (Precision * Recall) / (Precision + Recall)
```

### Argument Evaluation

Exact match por tipo de argumento:
- Participants match exato
- Time match exato
- Location match exato
- Topic match exato

### Overall Metrics

- **Exact Match Accuracy**: % de anotações 100% corretas (intent + todos arguments)
- **Confusion Matrix**: Erros de classificação por intent
- **Error Analysis**: Breakdown de FP, FN, missing, extra

## Performance

### Velocidade

- Processamento: ~100-150 emails/segundo
- Memória: ~50MB para 1000 emails
- Tamanho output: ~1MB para 1000 anotações JSON

### Limitações Conhecidas

1. **Triggers simples**: Pattern matching não captura contexto
   - Melhoria: Adicionar heurísticas de contexto
   
2. **Participantes**: Baixa precisão com nomes sem título
   - Melhoria: Usar NER com spaCy
   
3. **Temporal**: Não resolve referências absolutas
   - Melhoria: Temporal normalization com contexto de date
   
4. **Locations**: Não detecta endereços completos
   - Melhoria: Gazetteers de locations

5. **Topics**: Lexicon limitado
   - Melhoria: Expandir com embeddings

## Configurações Avançadas

### Alterar Triggers

```python
# Em heuristic_extractors.py
class TriggerExtractor:
    TRIGGER_PATTERNS = {
        'seu_novo_trigger': [
            r'\bpadrão1\b',
            r'\bpadrão2\b',
        ]
    }
```

### Customizar Validação

```python
# Em validators.py
class AnnotationValidator:
    VALID_INTENTS = [
        'seu_intent',
    ]
    ARGUMENT_FIELDS = [
        'seu_argumento',
    ]
```

### Ajustar Confidence Scores

```python
# Em validators.py normalize_annotation()
normalized['confidence'] = {
    'trigger': 0.95,      # Aumentar confiança
    'participants': 0.80,
    ...
}
```

## Integração com Modelos

### Formato compatível com spaCy

```python
# Gerar DocBin para treinamento spaCy
from spacy.training import DocBin
from spacy.tokens import Doc

doc_bin = DocBin()
for ann in annotations:
    # Converter para Doc com entities
    doc = create_spacy_doc(ann)
    doc_bin.add(doc)

doc_bin.to_disk("./training_data.spacy")
```

### Formato compatível com HuggingFace

```python
# Converter para dataset HF
from datasets import Dataset

dataset = Dataset.from_dict({
    'text': [ann['text'] for ann in annotations],
    'label': [ann['intent'] for ann in annotations],
    'entities': [...],
})

dataset.push_to_hub("seu_dataset")
```

## Troubleshooting Técnico

### Erro: UnicodeEncodeError

**Causa**: PowerShell Windows usa cp1252 por padrão

**Solução:**
```python
# Forçar UTF-8
import sys
sys.stdout.reconfigure(encoding='utf-8')
```

### Erro: FileNotFoundError

**Causa**: Caminho relativo do dataset não encontrado

**Solução:**
```bash
# Usar caminhos absolutos
python gold_annotations_generator.py /caminho/completo/input.json output.json

# Ou cd para diretório correto
cd gold_annotations
python gold_annotations_generator.py ../dataset/emails.json output.json
```

### Erro: list is not hashable

**Causa**: Usando lista como chave de dicionário

**Solução**: Já foi corrigido no código principal

## Contribuindo

### Adicionar novo extractor

1. Criar nova classe herdando padrão
2. Implementar `extract()` retornando `ExtractionResult`
3. Adicionar testes em `test_gold_annotations.py`
4. Documentar em README.md

### Exemplo de novo extractor

```python
class CustomExtractor:
    """Extrai informação customizada"""
    
    PATTERNS = {...}
    
    def extract(self, text: str) -> ExtractionResult:
        """Implementação do algoritmo"""
        values = []
        # Lógica de extração
        return ExtractionResult(
            values=values,
            confidence=0.8,
            metadata={'tipo': 'custom'}
        )
```

## Publicações Relacionadas

Este sistema implementa técnicas comuns em:

1. **Information Extraction**
   - Shallow parsing com regex
   - Padrão-based extraction
   
2. **Named Entity Recognition (NER)**
   - Salutation detection
   - Proper noun extraction
   
3. **Temporal Normalization**
   - Relative time expressions
   - Temporal anchoring
   
4. **Text Classification Evaluation**
   - Precision/Recall/F1 metrics
   - Confusion matrices

---

**Versão:** 1.0  
**Data:** 11 de Maio de 2024  
**Mantido por:** Email Recognition PT-PT Project
