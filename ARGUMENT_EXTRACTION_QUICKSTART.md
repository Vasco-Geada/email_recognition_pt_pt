# 🎯 Quick Start Guide - Argument Extraction

## Installation

Your environment already has all required dependencies:
- ✅ `spacy==3.8.13`
- ✅ `pt_core_news_sm==3.8.0` (Portuguese model)
- ✅ Standard library modules: `re`, `logging`, `dataclasses`

## Basic Usage

### 1️⃣ Initialize Extractor

```python
from preprocessing.argument_extraction import ArgumentExtractor

# Initialize (uses cached model on subsequent calls)
extractor = ArgumentExtractor(model_name="pt_core_news_sm")
```

### 2️⃣ Extract Arguments from Email

```python
# Email data (from your email classifier)
email_body = """
Gostaria de agendar uma reunião com o João Silva para discutir 
o projeto de IA. Podemos fazer amanhã às 15h na sala 203?
"""

email_subject = "Reunião - Projeto de IA"
predicted_intent = "agendamento_reuniao"  # From intent classifier
trigger = "agendar"  # From trigger extractor (optional)

# Extract all arguments
result = extractor.extract_with_context(
    email_body=email_body,
    email_subject=email_subject,
    predicted_intent=predicted_intent,
    trigger=trigger,
    include_confidence=True,  # Include confidence scores
)
```

### 3️⃣ Access Results

```python
# Get summary (just text values)
arguments_summary = result["extracted_arguments"]

# participants: ["João Silva"]
# time_expressions: ["amanhã", "às 15h"]
# locations: ["sala 203"]
# topics: ["projeto de IA"]

# Get metadata
metadata = result["metadata"]
# {
#   "email_subject": "Reunião - Projeto de IA",
#   "predicted_intent": "agendamento_reuniao",
#   "trigger": "agendar",
#   "extraction_timestamp": "2024-04-09T14:30:00.000000"
# }
```

---

## API Reference

### ArgumentExtractor

Main class for extracting arguments from Portuguese emails.

**Constructor:**
```python
ArgumentExtractor(model_name: str = "pt_core_news_sm")
```

**Methods:**

#### `extract()`
Extract arguments from email.

```python
arguments = extractor.extract(
    email_body: str,
    email_subject: str = "",
    predicted_intent: str = "",
    trigger: str = "",
) -> ExtractedArguments
```

**Returns:** `ExtractedArguments` object with:
- `.participants: List[ArgumentSpan]`
- `.time_expressions: List[ArgumentSpan]`
- `.locations: List[ArgumentSpan]`
- `.topics: List[ArgumentSpan]`

#### `extract_with_context()`
Extract arguments and return with metadata.

```python
result = extractor.extract_with_context(
    email_body: str,
    email_subject: str = "",
    predicted_intent: str = "",
    trigger: str = "",
    include_confidence: bool = True,
) -> Dict
```

**Returns:**
```python
{
    'extracted_arguments': {
        'participants': [...],
        'time_expressions': [...],
        'locations': [...],
        'topics': [...]
    },
    'metadata': {
        'email_subject': str,
        'predicted_intent': str,
        'trigger': str,
        'extraction_timestamp': str
    }
}
```

### ArgumentSpan

Represents extracted argument with metadata.

```python
@dataclass
class ArgumentSpan:
    text: str                    # Exact text from email
    span_start: int              # Character position start
    span_end: int                # Character position end
    confidence: float = 1.0      # 0-1 confidence score
    extraction_method: str = ""  # Method used: 'ner', 'regex', 'heuristic'
    
    def to_dict(self) -> Dict:   # Serialize to dictionary
        ...
```

---

## Component Extractors

### TemporalExpressionExtractor

Extracts Portuguese temporal expressions (dates, times, relative expressions).

```python
from preprocessing.argument_extraction import TemporalExpressionExtractor

extractor = TemporalExpressionExtractor()
spans = extractor.extract(email_text)
# Returns: List[ArgumentSpan] with temporal expressions
```

**Handles:**
- Specific dates: "5 de março", "mar 05 2024"
- Weekdays: "segunda", "sexta-feira", "sex."
- Relative dates: "amanhã", "próxima semana", "daqui a 3 dias"
- Times: "15h", "15:00", "às 15 horas"
- Informal: "de manhã", "depois de almoço", "de tarde"

### LocationExtractor

Extracts room/building references and physical locations.

```python
from preprocessing.argument_extraction import LocationExtractor

extractor = LocationExtractor()
spans = extractor.extract(email_text)
# Returns: List[ArgumentSpan] with locations
```

**Handles:**
- Rooms: "sala 203", "escritório nº 5", "auditório A"
- Floors: "1º andar", "piso 2"
- Named places: "lab de IA", "biblioteca", "cantina"
- Addresses: "Rua da Prata, 50"

### ParticipantExtractor

Extracts person names and emails.

```python
from preprocessing.argument_extraction import ParticipantExtractor
import spacy

nlp = spacy.load("pt_core_news_sm")
extractor = ParticipantExtractor(nlp_model=nlp)
spans = extractor.extract(email_text)
# Returns: List[ArgumentSpan] with participants
```

**Methods:**
- spaCy NER for named entities (confidence: 0.80)
- Regex for email addresses (confidence: 0.95)
- Pattern-based heuristics (confidence: 0.60)

### TopicExtractor

Extracts meeting topics using noun chunks and keywords.

```python
from preprocessing.argument_extraction import TopicExtractor
import spacy

nlp = spacy.load("pt_core_news_sm")
extractor = TopicExtractor(nlp_model=nlp)
spans = extractor.extract(email_text, subject="", intent="")
# Returns: List[ArgumentSpan] with topics (top 5 ranked)
```

**Strategies:**
- Noun chunk extraction
- Predefined topic keywords
- Intent-based context

---

## Integration with Existing Pipeline

### Full Email Processing Pipeline

```python
from preprocessing.argument_extraction import ArgumentExtractor
from models.predict_intent import predict_intent_from_text
from preprocessing.trigger_extraction import TriggerExtractor

# 1. Load necessary components
intent_predictor = predict_intent_from_text  # From your models
trigger_extractor = TriggerExtractor()
argument_extractor = ArgumentExtractor()

# 2. Process email
email_body = "..."
email_subject = "..."

# 3. Predict intent
intent_result = intent_predictor(email_body, email_subject)
predicted_intent = intent_result["intent"]

# 4. Extract trigger
trigger_result = trigger_extractor.extract_trigger(
    text=email_body,
    intent=predicted_intent
)
trigger = trigger_result.get("trigger", "") if trigger_result else ""

# 5. Extract arguments
arguments = argument_extractor.extract(
    email_body=email_body,
    email_subject=email_subject,
    predicted_intent=predicted_intent,
    trigger=trigger,
)

# 6. Combine results
structured_output = {
    "intent": predicted_intent,
    "trigger": trigger,
    "participants": arguments.summary()["participants"],
    "times": arguments.summary()["time_expressions"],
    "locations": arguments.summary()["locations"],
    "topics": arguments.summary()["topics"],
}
```

---

## Examples

### Example 1: Simple Extraction

```python
from preprocessing.argument_extraction import ArgumentExtractor

extractor = ArgumentExtractor()

email = "Reunião com Maria amanhã às 10h na sala 203"
result = extractor.extract(email)

print(result.summary())
# {
#   'participants': ['Maria'],
#   'time_expressions': ['amanhã', 'às 10h'],
#   'locations': ['sala 203'],
#   'topics': []
# }
```

### Example 2: With Confidence Scores

```python
result = extractor.extract_with_context(
    email_body=email,
    include_confidence=True,
)

for arg in result["extracted_arguments"]["participants"]:
    print(f"{arg['text']} (confidence: {arg['confidence']:.2f})")
# Maria (confidence: 0.85)
```

### Example 3: Handling Multiple Temporal Expressions

```python
email = "Reunião próxima segunda de tarde ou quinta-feira de manhã"
result = extractor.extract(email)

print([s.text for s in result.time_expressions])
# ['próxima segunda', 'de tarde', 'quinta-feira', 'de manhã']
```

### Example 4: Topic Extraction with Intent

```python
email = "Discussão sobre cronograma do Q4 e recursos da equipa"
result = extractor.extract(
    email_body=email,
    email_subject="Reunião de planejamento",
    predicted_intent="reuniao_confirmada",
)

topics = [s.text for s in result.topics[:3]]  # Top 3
print(topics)
# ['cronograma do Q4', 'recursos', 'equipa']
```

---

## Handling Special Cases

### Case 1: Extracting from Email Signature

```python
# Problem: Signature has many names that aren't meeting participants

email_with_signature = """
Reunião com João amanhã?

--
Maria Silva
Senior Engineer
Company XYZ
Phone: +351-21-1234567
Email: maria@company.com
Team: Paulo, Ana, Bruno (CC)
"""

# Solution: Pre-process to remove signature
import re

def remove_signature(text):
    # Remove everything after common signature markers
    signature_markers = [r'--\s*$', r'___+$', r'Enviado via', r'Enviado por']
    for marker in signature_markers:
        text = re.split(marker, text, flags=re.MULTILINE)[0]
    return text.strip()

clean_email = remove_signature(email_with_signature)
result = extractor.extract(clean_email)
# Only "João" extracted, not signature names
```

### Case 2: Handling Abbreviations

```python
# Problem: "Reunião com JS e MPS" - abbreviations not recognized

email = "Reunião com JS e MPS para discutir o projeto"

# Workaround 1: Add context dictionary
company_directory = {
    "JS": "João Silva",
    "MPS": "Maria Pereira Silva",
}

result = extractor.extract(email)
# Current: No names extracted

# Workaround 2: Pre-expand abbreviations
def expand_abbreviations(text, directory):
    for abbrev, full_name in directory.items():
        # Only expand if followed by space or punctuation
        text = re.sub(rf'\b{abbrev}\b', full_name, text)
    return text

expanded = expand_abbreviations(email, company_directory)
result = extractor.extract(expanded)
# Now: ["João Silva", "Maria Pereira Silva"] extracted
```

### Case 3: Handling Negated Expressions

```python
# Problem: "Não amanhã, mas sim sexta"

email = "Não posso amanhã, mas sexta-feira é perfeito"

result = extractor.extract(email)
# Current: ["amanhã", "sexta-feira"] (both extracted)

# Workaround: Post-process with negation awareness
def filter_negated_spans(text, spans):
    """Remove spans that are preceded by negation."""
    negation_markers = ['não', 'nunca', 'jamais', 'nenhum']
    filtered = []
    
    for span in spans:
        # Look 15 chars before span for negation
        context_start = max(0, span.span_start - 15)
        context = text[context_start:span.span_start].lower()
        
        is_negated = any(marker in context for marker in negation_markers)
        
        if not is_negated:
            filtered.append(span)
    
    return filtered

result.time_expressions = filter_negated_spans(email, result.time_expressions)
# Now: Only ["sexta-feira"] kept
```

---

## Performance Tips

### 1. Reuse Extractor Instance

```python
# ❌ SLOW: Creates new instance per email
for email in emails:
    extractor = ArgumentExtractor()  # Don't do this!
    result = extractor.extract(email)

# ✅ FAST: Reuse instance
extractor = ArgumentExtractor()  # Create once
for email in emails:
    result = extractor.extract(email)
```

### 2. Batch Processing

```python
from concurrent.futures import ThreadPoolExecutor

def extract_batch(emails, num_workers=4):
    """Extract arguments from multiple emails in parallel."""
    extractor = ArgumentExtractor()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(
            lambda e: extractor.extract(e),
            emails
        ))
    return results

# Process 1000 emails in parallel
all_results = extract_batch(email_list, num_workers=8)
```

### 3. Component-Specific Extraction

```python
# If you only need specific arguments, use component extractors directly

temporal_extractor = TemporalExpressionExtractor()
times = temporal_extractor.extract(email_text)

location_extractor = LocationExtractor()
locs = location_extractor.extract(email_text)

# Faster than full extraction if you don't need all arguments
```

---

## Troubleshooting

### Error: "Model 'pt_core_news_sm' not found"

```bash
python -m spacy download pt_core_news_sm
```

### No Arguments Extracted

**Possible causes:**
1. **Text is English:** Module is Portuguese-specific
   - Solution: Add language detection first

2. **Text is too short:** Single-word emails
   - Solution: Combine with email subject for context

3. **Non-standard formatting:** Unusual email structure
   - Solution: Pre-process to normalize formatting

**Debugging:**
```python
text = "Some email text"
temporal = TemporalExpressionExtractor()
times = temporal.extract(text)
print(f"Regex matches: {times}")  # Check each component separately
```

### Low Confidence Scores

Confidence varies by extraction method:

```python
result = extractor.extract_with_context(email_text, include_confidence=True)

for arg_type, spans in result["extracted_arguments"].items():
    for span in spans:
        if span['confidence'] < 0.7:
            print(f"⚠️  Low confidence {arg_type}: '{span['text']}' ({span['confidence']})")
```

---

## Next Steps

1. **Evaluate on your data:** Run `python test_argument_extraction.py`
2. **Integrate into pipeline:** Add to `preprocessing/email_pipeline.py`
3. **Collect annotations:** Build evaluation set for metrics
4. **Iterate patterns:** Refine regex based on failures
5. **Upgrade to ML:** Transition to token classification when labeled data available

See **ARGUMENT_EXTRACTION_GUIDE.md** for advanced topics and evolution path.
