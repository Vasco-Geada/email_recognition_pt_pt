# Sistema de Gold Annotations para Email Recognition PT-PT

Sistema automático/semi-automático de geração, validação e avaliação de gold annotations para projeto de NLP em português europeu.

## 🎯 Visão Geral

Este sistema gera anotações estruturadas para emails académicos informais em português europeu, focado na detecção de intenções de reunião (agendamento, cancelamento, confirmação).

### Estrutura de Saída

```json
{
  "id": 1,
  "text": "Boas Ana, podemos reunir amanhã às 15h no Teams?",
  "intent": "agendamento_reuniao",
  "trigger": "reunir",
  "arguments": {
    "participants": ["Ana"],
    "time": ["amanhã às 15h"],
    "location": ["Teams"],
    "topic": []
  },
  "confidence": {
    "trigger": 0.9,
    "participants": 0.75,
    "temporal": 0.8,
    "location": 0.85,
    "topic": 0.0
  },
  "metadata": {
    "source_subject": "Reunião",
    "extracted_at": "2024-05-11T10:30:00",
    "extraction_method": "heuristic"
  }
}
```

## 📁 Estrutura de Ficheiros

```
gold_annotations/
├── heuristic_extractors.py      # Extractores heurísticos
├── validators.py                 # Validadores
├── gold_annotations_generator.py # Orquestrador principal
├── evaluate_annotations.py       # Avaliador
├── example_usage.py              # Exemplos
├── README.md                      # Este ficheiro
├── output/                        # Saídas (criado automaticamente)
└── __init__.py
```

## 🚀 Quick Start

### Instalação

```bash
# Nenhuma dependência externa além de stdlib Python
python --version  # 3.11+
```

### Uso Básico

#### 1. Gerar Gold Annotations

```bash
python gold_annotations_generator.py input.json output.json
```

**Input esperado** (`input.json`):
```json
[
  {
    "subject": "Reunião amanhã",
    "body": "Boas Ana, podemos reunir amanhã às 15h?",
    "label": "agendamento_reuniao"
  }
]
```

**Output gerado** (`output.json`):
Anotações estruturadas com triggers, participantes, expressões temporais, localizações e tópicos.

#### 2. Validar Anotações

```python
from validators import AnnotationValidator

validator = AnnotationValidator()
result = validator.validate_batch(annotations)

if result.is_valid:
    print("✓ Anotações válidas")
else:
    for error in result.errors:
        print(f"✗ {error.error_type}: {error.message}")
```

#### 3. Avaliar Predições

```bash
python evaluate_annotations.py gold.json predictions.json -o report.json
```

Produz:
- Precision, Recall, F1 por classe
- Exact Match accuracy
- Matriz de confusão
- Análise de erros

### Exemplos

Executar todos os exemplos:

```bash
python example_usage.py
```

Isto demonstra:
1. Geração de gold annotations
2. Carregamento e inspeção
3. Revisão manual (simulada)
4. Validação
5. Avaliação

## 📊 Componentes

### 1. **heuristic_extractors.py**

Extração heurística baseada em padrões regex:

#### TriggerExtractor
- Detecta verbos relacionados com reunião
- Triggers: reunir, marcar, cancelar, confirmar, agendar, combinar, encontrar, falar, discutir

```python
from heuristic_extractors import TriggerExtractor

extractor = TriggerExtractor()
result = extractor.extract("Vamos reunir amanhã?")
print(result.values)  # ['reunir']
print(result.confidence)  # 0.9
```

#### ParticipantExtractor
- Detecta nomes próprios
- Contextos: salutações (Olá Ana), títulos (Prof. Silva), menções (@nome)

#### TemporalExtractor
- Expressões temporais relativas: amanhã, sexta, próxima semana
- Horas: 15h, às 15:00, meio-dia
- Períodos: manhã, tarde, depois de almoço

#### LocationExtractor
- Plataformas online: Teams, Zoom, Discord
- Salas: sala 2.3, auditório
- Espaços: biblioteca, laboratório

#### TopicExtractor
- Trabalhos académicos: dissertação, tese, relatório
- Conceitos técnicos: NLP, BERT, pipeline, F1
- Disciplinas: linguística, IA

### 2. **validators.py**

Validação de integridade e consistência:

#### AnnotationValidator
- Valida campos obrigatórios
- Verifica tipos corretos
- Intents válidos: agendamento_reuniao, cancelamento_reuniao, reuniao_confirmada
- Normalização de anotações

```python
from validators import AnnotationValidator

validator = AnnotationValidator()
normalized = validator.normalize_annotation(annotation)
result = validator.validate_single_annotation(normalized)
```

#### JSONValidator
- Validação de ficheiros JSON
- Verificação UTF-8

#### ConsistencyValidator
- Consistência intent-trigger
- Avisos de anomalias

### 3. **gold_annotations_generator.py**

Orquestrador principal do pipeline:

```python
from gold_annotations_generator import GoldAnnotationsGenerator

generator = GoldAnnotationsGenerator(verbose=True)
success = generator.run('emails.json', 'gold_annotations.json')
```

Pipeline:
1. Carrega emails JSON
2. Processa cada email
3. Extrai features heuristicamente
4. Valida anotações
5. Normaliza output
6. Salva JSON
7. Gera relatório

### 4. **evaluate_annotations.py**

Avaliação científica de anotações:

```bash
python evaluate_annotations.py gold.json predictions.json -o report.json
```

Métricas:
- **Exact Match**: Coincidência total (intent + arguments)
- **Intent Accuracy**: Apenas intent
- **Precision/Recall/F1**: Por classe de intent
- **Argument Metrics**: Por tipo (participants, time, location, topic)
- **Confusion Matrix**: Erros de classificação
- **Error Analysis**: Desagregação de erros

## 🔧 Configuração Avançada

### Customizar Triggers

```python
# Em heuristic_extractors.py
class TriggerExtractor:
    TRIGGER_PATTERNS = {
        'seu_trigger': [
            r'\bpadrão1\b',
            r'\bpadrão2\b',
        ]
    }
```

### Customizar Localizações

```python
# Em heuristic_extractors.py
class LocationExtractor:
    LOCATION_PATTERNS = {
        'seu_tipo': [
            r'\bpadrão\b',
        ]
    }
```

### Customizar Intents

```python
# Em validators.py
class AnnotationValidator:
    VALID_INTENTS = [
        'seu_intent_1',
        'seu_intent_2',
    ]
```

## 📈 Workflow Recomendado

### 1. Geração Automática

```bash
python gold_annotations_generator.py raw_emails.json auto_annotations.json
```

### 2. Revisão Manual

Editar `auto_annotations.json` manualmente:
- Corrigir triggers mal detectados
- Adicionar participantes omitidos
- Refinar expressões temporais
- Marcar como revisado em metadata

### 3. Validação

```python
from validators import AnnotationValidator

validator = AnnotationValidator()
result = validator.validate_batch(reviewed_annotations)
```

### 4. Exportar para Avaliação

```bash
cp reviewed_annotations.json gold_annotations.json
```

### 5. Avaliar Modelo

```bash
python evaluate_annotations.py gold_annotations.json model_predictions.json
```

## 🎓 Boas Práticas para NLP Académico

### 1. Anotação Consistente

- Usar as mesmas formas/variações para mesmo conceito
- Manter lista de sinónimos
- Documentar decisões de anotação

### 2. Controlo de Qualidade

- Validar 100% das anotações
- Verificar duplicados
- Verificar UTF-8 encoding
- Manter changelog de revisões

### 3. Rastreabilidade

- Manter `metadata` com informações de origem
- Registar versões de gold annotations
- Documentar mudanças

### 4. Reprodutibilidade

- Usar seeds fixos em operações aleatórias
- Documentar parâmetros de configuração
- Versionar código de extração

### 5. Relatórios

- Sempre gerar relatório de processamento
- Documentar taxa de sucesso
- Analisar distribuição de classes
- Investigar erros sistemáticos

## 📋 Exemplos de Uso

### Exemplo 1: Pipeline Completo

```python
from gold_annotations_generator import GoldAnnotationsGenerator
from validators import AnnotationValidator
from evaluate_annotations import AnnotationEvaluator

# Gerar
generator = GoldAnnotationsGenerator()
emails = generator.load_emails_json("emails.json")
annotations, errors = generator.process_batch(emails)
normalized, msg = generator.validate_annotations(annotations)
generator.save_annotations(normalized, "gold.json")

# Validar
validator = AnnotationValidator()
result = validator.validate_batch(normalized)

# Avaliar (com predições de modelo)
evaluator = AnnotationEvaluator()
eval_result = evaluator.evaluate_batch(normalized, predictions)
evaluator.print_metrics(eval_result)
```

### Exemplo 2: Revisão Manual

```python
import json
from validators import AnnotationValidator

# Carregar anotações geradas
with open('auto_annotations.json') as f:
    annotations = json.load(f)

# Revisar e editar (interativamente)
for ann in annotations:
    print(f"\nAnotação {ann['id']}: {ann['text'][:50]}...")
    print(f"  Intent: {ann['intent']}")
    print(f"  Trigger: {ann['trigger']}")
    
    # Editar manualmente conforme necessário
    # ann['trigger'] = input("Trigger corrigido: ") or ann['trigger']
    
    # Marcar como revisado
    ann['metadata']['reviewed'] = True
    ann['metadata']['reviewer'] = 'seu_nome'

# Validar e salvar
validator = AnnotationValidator()
normalized = [validator.normalize_annotation(a) for a in annotations]
result = validator.validate_batch(normalized)

with open('reviewed_annotations.json', 'w', encoding='utf-8') as f:
    json.dump(normalized, f, indent=2, ensure_ascii=False)
```

### Exemplo 3: Extração Customizada

```python
from heuristic_extractors import HeuristicAnnotationExtractor

extractor = HeuristicAnnotationExtractor()

email_text = """
Oi Professor João,

Podemos marcar uma reunião para segunda-feira à tarde no Zoom 
para falarmos do pipeline BERT e dos resultados da dissertação?

Tenho disponibilidade a partir das 14h.

Obrigado,
Ana
"""

result = extractor.extract_all(email_text)

print("Trigger:", result['trigger'].values)
print("Participants:", result['participants'].values)
print("Time:", result['temporal'].values)
print("Location:", result['location'].values)
print("Topic:", result['topic'].values)
```

## 🐛 Troubleshooting

### Problema: Poucas anotações com sucesso

**Solução:**
1. Verificar formato de input JSON
2. Aumentar confiança de extractores
3. Adicionar novos padrões regex

### Problema: Erros de encoding

**Solução:**
```python
# Garantir UTF-8
with open('file.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
```

### Problema: Triggers não detectados

**Solução:**
1. Adicionar novo trigger a `TriggerExtractor.TRIGGER_PATTERNS`
2. Testar padrão regex
3. Aumentar cobertura de variações

### Problema: Falsos positivos em participantes

**Solução:**
1. Aumentar comprimento mínimo de nome
2. Refinar padrões de salutação
3. Adicionar lista de exclusões (pronomes, palavras comuns)

## 📚 Referências Técnicas

### Formatos Suportados

- **Input**: JSON com campos `subject`, `body`, `label`
- **Output**: JSON com estrutura padrão (ver seção anterior)
- **Encoding**: UTF-8 obrigatório

### Compatibilidade

- Python 3.11+
- Sem dependências externas (apenas stdlib)
- Cross-platform (Windows, Linux, macOS)

### Performance

- Processamento típico: ~100 emails/segundo
- Memória: ~50MB para 1000 emails
- Discos: ~1MB para 1000 anotações JSON

## 📝 Changelog

### v1.0 (2024-05-11)
- ✓ Geração automática de gold annotations
- ✓ Extração heurística de triggers, participantes, temporal, location, topic
- ✓ Validação e normalização
- ✓ Avaliação com métricas científicas
- ✓ Suporte para revisão manual
- ✓ Documentação completa
- ✓ Exemplos de uso

## 🤝 Contribuições

Para adicionar novos recursos:

1. Adicionar novos padrões ao extractor relevante
2. Atualizar validador se necessário
3. Adicionar testes em `example_usage.py`
4. Documentar em `README.md`

## 📄 Licença

Este código é fornecido como parte do projeto Email Recognition PT-PT.

## ✉️ Suporte

Para dúvidas ou sugestões, consulte a documentação ou execute:

```bash
python example_usage.py
```

---

**Última atualização:** 11 de maio de 2024
