# QUICKSTART - Sistema de Gold Annotations

Guia rápido para começar a gerar gold annotations para seu projeto NLP em português.

## ⚡ 5 Minutos para Começar

### 1. Preparar o Dataset

Crie um ficheiro `emails.json` com seus emails:

```json
[
  {
    "subject": "Reunião amanhã",
    "body": "Boas Ana, podemos reunir amanhã às 15h no Teams para discutir o dataset?",
    "label": "agendamento_reuniao"
  },
  {
    "subject": "Cancelar reunião",
    "body": "Infelizmente não consigo aparecer sexta. Podes marcar para próxima semana?",
    "label": "cancelamento_reuniao"
  },
  {
    "subject": "Reunião confirmada",
    "body": "Confirmed! Nos vemos sexta às 14h na biblioteca para falarmos do pipeline.",
    "label": "reuniao_confirmada"
  }
]
```

### 2. Gerar Gold Annotations

```bash
python gold_annotations_generator.py emails.json gold_annotations.json
```

Output: `gold_annotations.json` com anotações estruturadas

### 3. Revisar (Opcional)

Editar `gold_annotations.json` manualmente para corrigir erros:

```json
{
  "id": 1,
  "text": "Boas Ana, podemos reunir amanhã às 15h no Teams para discutir o dataset?",
  "intent": "agendamento_reuniao",
  "trigger": "reunir",
  "arguments": {
    "participants": ["Ana"],
    "time": ["amanhã às 15h"],
    "location": ["Teams"],
    "topic": ["dataset"]
  },
  "confidence": {
    "trigger": 0.9,
    "participants": 0.75,
    "temporal": 0.8,
    "location": 0.85,
    "topic": 0.5
  }
}
```

### 4. Validar

```bash
python -c "
from validators import AnnotationValidator
import json

with open('gold_annotations.json') as f:
    anns = json.load(f)

validator = AnnotationValidator()
result = validator.validate_batch(anns)

print(f'Válido: {result.is_valid}')
print(f'Erros: {len(result.errors)}')
print(f'Avisos: {len(result.warnings)}')
"
```

### 5. Avaliar (com Predições)

```bash
python evaluate_annotations.py gold_annotations.json predictions.json -o report.json
```

## 📋 Checklist de Início

- [ ] Python 3.11+ instalado
- [ ] Ficheiro `emails.json` preparado
- [ ] Executar `python gold_annotations_generator.py emails.json output.json`
- [ ] Revisar `output.json` manualmente
- [ ] Validar com `test_gold_annotations.py`
- [ ] Pronto para avaliação!

## 🎯 Próximos Passos

1. **Integrar com seu modelo NLP**
   ```python
   from gold_annotations import GoldAnnotationsGenerator
   
   gen = GoldAnnotationsGenerator()
   anns, errors = gen.process_batch(emails)
   ```

2. **Usar para avaliação científica**
   ```bash
   python evaluate_annotations.py gold.json model_output.json -o metrics.json
   ```

3. **Customizar extractores**
   - Adicionar novos triggers em `heuristic_extractors.py`
   - Ajustar padrões regex
   - Testar com `test_gold_annotations.py`

## 📚 Exemplos Completos

Executar demonstração completa:

```bash
python example_usage.py
```

Isto cria:
- `output/gold_annotations_v1.json` - Anotações geradas
- `output/gold_annotations_reviewed.json` - Anotações revisadas
- `output/predictions_v1.json` - Predições simuladas
- `output/evaluation_report.json` - Relatório de avaliação

## 🔍 Verificar Extractores

Testar cada extractor individualmente:

```python
from heuristic_extractors import (
    TriggerExtractor, ParticipantExtractor, TemporalExtractor,
    LocationExtractor, TopicExtractor
)

text = "Boas Ana, podemos reunir amanhã às 15h no Teams?"

print("Triggers:", TriggerExtractor().extract(text).values)
print("Participantes:", ParticipantExtractor().extract(text).values)
print("Temporal:", TemporalExtractor().extract(text).values)
print("Localizações:", LocationExtractor().extract(text).values)
print("Tópicos:", TopicExtractor().extract(text).values)
```

## ✅ Validar Tudo

```bash
python test_gold_annotations.py
```

Resultado esperado:
```
[TEST] TriggerExtractor - Triggers Básicos
  ✓ 'Vamos reunir amanhã' -> reunir
  ...

[TEST] AnnotationValidator - Anotação Válida
  ✓ Anotação válida
  ...

✓ TESTES CONCLUÍDOS
```

## 🐛 Problemas Comuns

### Erro: `ModuleNotFoundError`
Certifique-se de que está no diretório `gold_annotations/`:
```bash
cd gold_annotations
python gold_annotations_generator.py ...
```

### Erro: `FileNotFoundError`
Ficheiro JSON não encontrado:
```bash
# Verificar se ficheiro existe
ls emails.json

# Usar caminho completo se necessário
python gold_annotations_generator.py /caminho/completo/emails.json output.json
```

### Poucas anotações encontradas
Adicionar mais padrões regex em `heuristic_extractors.py`:
```python
class TriggerExtractor:
    TRIGGER_PATTERNS = {
        'seu_trigger': [r'\bpadrão\b']
    }
```

## 📊 Estrutura de Output

Cada anotação contém:

```json
{
  "id": 1,                          // ID sequencial
  "text": "...",                    // Texto do email
  "intent": "agendamento_reuniao",  // Classe (3 opções)
  "trigger": "reunir",              // Verbo principal
  "arguments": {
    "participants": ["Ana"],        // Nomes mencionados
    "time": ["amanhã às 15h"],     // Expressões temporais
    "location": ["Teams"],          // Localizações
    "topic": []                     // Tópicos académicos
  },
  "confidence": {
    "trigger": 0.9,                // Confiança por campo
    "participants": 0.75,
    "temporal": 0.8,
    "location": 0.85,
    "topic": 0.0
  },
  "metadata": {
    "source_subject": "...",
    "extracted_at": "...",
    "extraction_method": "heuristic"
  }
}
```

## 🚀 Próximo Nível

### Exportar para spaCy (Bónus)

```python
import json
from spacy.training import Example
from spacy.tokens import Doc

# Carregar anotações
with open('gold_annotations.json') as f:
    annotations = json.load(f)

# Converter para formato spaCy DocBin
# (Implementação customizada necessária)
```

### Adicionar BIO Tagging (Bónus)

```python
def convert_to_bio(annotation):
    """Converte para BIO tagging"""
    text = annotation['text']
    bio_tags = ['O'] * len(text.split())
    
    # Marcar participantes
    for participant in annotation['arguments']['participants']:
        # Lógica BIO here
        pass
    
    return bio_tags
```

## 📞 Suporte

Consulte ficheiros:
- `README.md` - Documentação completa
- `example_usage.py` - Exemplos funcionais
- `test_gold_annotations.py` - Testes

---

**Pronto!** Agora pode começar a gerar suas gold annotations. 🎉
