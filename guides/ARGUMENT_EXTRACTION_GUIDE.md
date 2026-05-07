# 📊 Argument Extraction Strategy & Implementation Guide

## Overview

This guide explains the baseline architecture for extracting structured arguments (participants, time, location, topic) from Portuguese meeting emails. The system uses a **modular, multi-strategy approach** combining:

- **spaCy NER** for participant extraction
- **Regex patterns** for temporal and location detection
- **Lexical heuristics** for topic extraction

## System Architecture

```
Email Input
    ↓
[Subject + Body + Intent + Trigger]
    ↓
    ├─→ TemporalExpressionExtractor (regex patterns)
    ├─→ LocationExtractor (regex + heuristics)
    ├─→ ParticipantExtractor (spaCy NER + patterns)
    └─→ TopicExtractor (noun chunks + keywords)
    ↓
ExtractedArguments
(participants, times, locations, topics with spans)
```

---

## 1. Temporal Expression Extraction Strategy

### Why Pattern-Based Approach?

Portuguese meeting emails contain diverse temporal expressions that vary widely:

```
✓ Specific dates:      "5 de março", "próxima segunda"
✓ Relative dates:      "amanhã", "semana que vem"
✓ Time of day:         "às 15h", "depois de almoço"
✓ Informal:            "fim de semana", "de tarde"
✓ Combinations:        "quinta-feira às 10h"
```

**Why Regex?** These patterns follow consistent linguistic rules in Portuguese. Tokens-based approaches (e.g., BERT token classification) would require labeled data, whereas regex patterns can capture these expressions without training data.

### Pattern Categories

#### 1.1 Specific Dates
```python
# Matches: "5 de março", "15 de janeiro", "30 de dezembro de 2024"
r'\b\d{1,2}\s+de\s+(?:janeiro|fevereiro|...)\s*(?:de\s+\d{4})?\b'

# Matches: "mar 05", "mar 05 2024" (abbreviated form)
r'\b(?:jan|fev|mar|...)\s+\d{1,2}(?:\s+\d{4})?\b'

# Matches: "05/03", "05-03-2024" (numerical)
r'\b\d{1,2}[-/]\d{1,2}(?:[-/]\d{2,4})?\b'
```

#### 1.2 Weekdays
```
Monday:    segunda-feira, segunda, seg., 2ª
Tuesday:   terça-feira, terça, ter., 3ª
...Friday: sexta-feira, sexta, sex., 6ª

Pattern: r'\b(?:segunda|seg)\.?\s*-?\s*(?:feira)?\b'
(Case-insensitive, handles abbreviations and hyphens)
```

#### 1.3 Relative Temporal Expressions

| Expression | Pattern | Examples |
|------------|---------|----------|
| Tomorrow | `amanhã` | "Reunião amanhã" |
| Today | `hoje`, `hj` | "Hoje às 3pm" |
| This week | `esta semana` | "Monday próximo esta semana" |
| Next week | `próxima semana` | "Reunião próxima semana" |
| In X days | `daqui a {num} dias` | "Daqui a 3 dias" |

#### 1.4 Informal Time Expressions (Critical for Portuguese!)

Portuguese emails frequently use informal time markers:

| Expression | Pattern | Meaning |
|------------|---------|---------|
| De manhã | `de\s+(?:manhã)` | In the morning |
| Após almoço | `(?:depois\|após)\s+de\s+almoço` | After lunch |
| Fim de semana | `fim\s+de\s+semana` | Weekend |
| De tarde | `de\s+tarde` | In the afternoon |
| À noite | `à\s+noite` | In the evening |

**Why challenging?** These are highly context-dependent and vary by speaker. Examples:
- "Depois de almoço" can mean 1PM-5PM depending on context
- "De manhã" could be 8AM-12PM
- Requires post-processing with heuristics or learning

#### 1.5 Time of Day Patterns

```python
# Exact times
r'\b(?:às|as|a)\s+(?:\d{1,2}):?(?:\d{2})?\s*(?:h|horas)?\b'
# Matches: "às 15h", "as 15:00", "a 15 horas"

# 24-hour format
r'\b\d{1,2}:?\d{2}\s*(?:h|horas)?\b'
# Matches: "15:00", "15h", "15 horas"

# Hour only
r'\b(?:\d{1,2})\s*horas?\b'
# Matches: "15 horas", "3 horas"
```

### Temporal Expression Examples from Real Emails

```
Email 1: "Gostaria de agendar uma reunião para amanhã às 15h"
Output:  ["amanhã", "às 15h"]

Email 2: "Reunião próxima segunda-feira de tarde no escritório"
Output:  ["próxima segunda-feira", "de tarde"]

Email 3: "Podemos agendar para sexta 15:00?"
Output:  ["sexta", "15:00"]

Email 4: "Daqui a 3 dias para as 9 horas?"
Output:  ["Daqui a 3 dias", "9 horas"]

Email 5: "Depois de almoço amanhã, alguma sala disponível?"
Output:  ["Depois de almoço", "amanhã"]
```

### Handling Overlaps and Ambiguities

**Challenge:** Email: "Próxima semana segunda às 15h"
- Could extract: ["próxima semana segunda", "às 15h"] → overlapping
- Solution: Longest-match deduplication strategy

```python
def _deduplicate_spans(spans):
    """Remove overlapping spans, keeping longest matches"""
    sorted_spans = sorted(spans, key=lambda s: (s.span_start, -(s.span_end - s.span_start)))
    deduplicated = []
    for span in sorted_spans:
        has_overlap = any(
            not (span.span_end <= existing.span_start or span.span_start >= existing.span_end)
            for existing in deduplicated
        )
        if not has_overlap:
            deduplicated.append(span)
    return deduplicated
```

---

## 2. Location Extraction Strategy

### Pattern Categories

#### 2.1 Room/Space References
```python
# Room patterns
r'\b(?:sala|escritório|gabinete|auditório)\s+(?:de\s+)?(?:n°|nº|#)?(\d+|[A-Z])\b'
# Matches: "sala 203", "auditório A", "escritório nº 5"

# Floor patterns
r'\b\d+[°º]\s*(?:andar|piso)\b'
# Matches: "1º andar", "2º piso"

# Building blocks
r'\b(?:bloco|edifício)\s+[A-Z0-9]+\b'
# Matches: "bloco A", "edifício 1"
```

#### 2.2 Named Location Patterns
```python
# Institutional spaces
r'\b(?:auditório|laboratório|biblioteca|cafetaria|cantina|armazém)\b'

# Street addresses (basic)
r'\b(?:rua|avenida|av|praça)\s+[A-Z][a-záéíóúàâêõç\s]+\b'
```

### Location Examples

```
"Reunião na sala 203"          → "sala 203"
"Encontro no 1º andar"         → "1º andar"
"Auditório B às 15h"           → "Auditório B"
"Rua da Prata, 50"             → "Rua da Prata"
"Laboratório de IA"            → "Laboratório de IA"
"Bloco D, escritório 5"        → "Bloco D", "escritório 5"
```

### Context-Based Refinement (Future)

Locations can be ambiguous:
- "sala de reuniões" (meeting room) vs "sala de espera" (waiting room)
- Requires intent context: if `agendamento_reuniao`, prioritize meeting rooms

**Current:** Keep all matches with `.confidence = 0.85`
**Future:** Add intent-aware confidence adjustment

---

## 3. Participant Extraction Strategy

### Multi-Source Extraction

#### 3.1 spaCy NER (Primary)
```python
# Use pt_core_news_sm to identify PERSON entities
for ent in doc.ents:
    if ent.label_ == 'PER':
        extract_span(ent.text)
```

**Strengths:**
- Handles complex names with prefixes: "Dr. João Silva"
- Context-aware: understands names vs. common nouns
- ~85-90% accuracy on news text

**Weaknesses:**
- Not fine-tuned for meeting emails
- May miss informal references ("João" vs full name)
- Struggles with abbreviations ("JS" for João Silva)

#### 3.2 Email Address Extraction
```python
EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
```
High precision (95%+), extracts: addresses@company.com

#### 3.3 Pattern-Based Heuristics

For informal Portuguese:
```python
# "com o João" → João
r'(?:com|entre|por)\s+(?:o\s+|a\s+)([A-Z][a-záéíóúàâêõç\s]+?)(?:\s*(?:,|$))'

# "participar à reunião com Maria" → Maria
r'(?:participar|presença|estar|comparecer)\s+...([A-Z][a-záéíóúàâêõç\s]+?)'
```

**Challenges:**
- Case sensitivity: Names must start with capital letter
- Abbreviations: "JS", "JSA" are harder to detect
- Email signatures: Often list many names not relevant to meeting

### Confidence Scoring

| Method | Confidence | Reason |
|--------|-----------|--------|
| spaCy NER | 0.80 | Good but email-specific domain shift |
| Email address | 0.95 | Pattern is precise, but may not be participant |
| Pattern-based | 0.60 | High false positive rate in informal text |

### Participant Extraction Examples

```
Email: "Gostaria de agendar uma reunião com o Dr. Silva e João Costa"
Output: ["Dr. Silva", "João Costa"] (from NER)

Email: "Para confirmar com maria@company.com e pedro@company.com"
Output: ["maria@company.com", "pedro@company.com"] (from regex)

Email: "entre a Ana e o Bruno"
Output: ["Ana", "Bruno"] (from pattern heuristics)
```

---

## 4. Topic Extraction Strategy

### Multi-Strategy Approach

Topic extraction is the hardest argument type because topics are variable and domain-specific.

#### 4.1 Noun Chunk Extraction (spaCy)

```python
for chunk in doc.noun_chunks:
    if not_all_stopwords(chunk):
        confidence = len(content_words) / len(words)  # 0.5-0.9
        extract(chunk.text)
```

**Why noun chunks?** Meeting topics are typically noun phrases:
- "projeto de AI" (AI project)
- "demonstração do protótipo" (prototype demo)
- "orçamento do Q4" (Q4 budget)

**Challenges:**
- Over-extraction: "reunião de meeting" captures both words
- Conjunction handling: "email e website" = 2 topics vs 1?
- Salience: "a reunião" has low information content

#### 4.2 Keyword-Based Heuristics

Predefined topic keywords by category:

```python
TOPIC_KEYWORDS = {
    'projeto': ['projeto', 'desenvolvimento', 'code', 'sprint'],
    'orçamento': ['orçamento', 'preço', 'custo', 'valor'],
    'recursos': ['recursos', 'equipa', 'staff', 'pessoal'],
    'cronograma': ['cronograma', 'timeline', 'prazos', 'milestone'],
    'qualidade': ['qualidade', 'testes', 'qc', 'qa'],
    'apresentação': ['apresentação', 'demo', 'showcase'],
}
```

Matches keywords in email → assign topic category + confidence 0.7

#### 4.3 Intent-Based Context

Use predicted intent to weight topics:

```
Intent: agendamento_reuniao
→ Topic likely related to meeting content (not just scheduling)

Intent: cancelamento_reuniao
→ Topic may explain cancellation reason
```

### Limiting Output

Topics can be numerous. Ranking by confidence and limiting to top 3-5:

```python
return sorted(spans, key=lambda s: s.confidence, reverse=True)[:5]
```

### Topic Extraction Examples

```
Email: "Reunião para discutir o projeto de IA e análise de cronograma"
Noun chunks: ["reunião", "projeto de IA", "análise", "cronograma"]
Keywords:    ["projeto"] (from "projeto de IA"), ["cronograma"]
Output:      [
    ArgumentSpan("projeto de IA", confidence=0.85, method='noun_chunks'),
    ArgumentSpan("cronograma", confidence=0.75, method='keyword_heuristic')
]

Email: "Update sobre custos e recursos para Q4"
Output: [
    ArgumentSpan("custos", confidence=0.70, method='keyword'),
    ArgumentSpan("recursos", confidence=0.70, method='keyword'),
    ArgumentSpan("Q4", confidence=0.60, method='noun_chunks')
]
```

---

## 5. Failure Cases & Limitations

### Temporal Expressions

| Failure Case | Example | Root Cause | Mitigation |
|--------------|---------|-----------|------------|
| Implicit dates | "Confirmo para quinta" (Thursday) | Relies on current date context | Store extraction date; post-process to relative dates |
| Ambiguous time | "15h" (3 PM or 15:00 24h?) | Portuguese uses both formats | Assume 24h format, same as ISO standard |
| Informal only | "Na hora de almoço" | No specific time → heuristics | Generate time range "12:00-13:00" with low confidence |
| Duration confusion | "reunião de 1 hora" | May extract "1 hora" as time | Filter patterns that follow "de" (duration marker) |
| Context violations | "Não às 15h mas às 16h" | Regex captures both times | Add negation context checking |

### Locations

| Failure Case | Example | Root Cause | Mitigation |
|--------------|---------|-----------|------------|
| Ambiguous "sala" | "Discuta na sala de espera" | "sala" could be meeting room or generic space | Intent filtering: agendamento → assume meeting room |
| Multiple locations | "Sala 203, bloco A" | May extract as 2 locations | Merge related locations into compound spans |
| Address-like patterns | "Coluna 5, seção 2" | Could match spurious room patterns | Add whitelisting for known building layouts |
| Uninformative | "o local" (the location) | Pronoun without specific place | Filter generic references |

### Participants

| Failure Case | Example | Root Cause | Mitigation |
|--------------|---------|-----------|------------|
| Title as name | "Dr. Silva e Eng. Costa" | NER may miss second occurrence | Post-process to extract after title markers |
| Signature avalanche | 40 names in email footer | Signature parsing needed | Implement sender/recipient extraction instead |
| Abbreviations | "JSA e MPS" | No context for expansion | Add company directory lookup (future) |
| Gendered titles | "Senhora Paulo" (uncommon) | Heuristics assume name after title | Reduce confidence for unusual patterns |
| Email-only mention | "contact@example.com" (no name) | Can't resolve email to person | Store email as fallback; note in confidence |

### Topics

| Failure Case | Example | Root Cause | Mitigation |
|--------------|---------|-----------|------------|
| Generic topics | "reunião", "discussão" | Common words not informative | Add these to stop words |
| Missing context | "Novo projeto X" where X undefined | Needs external knowledge | Keep as topic; note uncertainty in confidence |
| Negative topics | "Não sobre orçamento" | Regex ignores negation | Add negation detection (simple: check "não" in context) |
| Acronyms | "AI", "ML", "CV" | Abbreviations without expansion | Require acronym expansion database |
| Multiple emails in one | Forwarded chains | Topics from multiple meetings mixed | Split by "---" or "From:" boundaries (preprocessing) |

---

## 6. Evaluation Metrics for Span Extraction

### Exact Match Evaluation (Strict)

```
Metric: F1 Score with exact span matching

Predicted span: "15h" at [10-13]
Reference span: "às 15h" at [8-13]
→ FALSE (different start position)
```

**Formula:**
```
Precision = (# exact match correct) / (# predicted)
Recall = (# exact match correct) / (# reference)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### Partial Match Evaluation (Relaxed)

Allow span boundaries to vary by N characters:

```
Predicted: "15h" [10-13]
Reference: "às 15h" [8-13]
Token-level overlap: "15h" matches → Consider as partial match

Tolerance: ±2 characters
```

### Category-Specific Metrics

#### Temporal Expressions
```
Question: What temporal values are normalized correctly?

Email: "Reunião amanhã"
Predicted: "amanhã"
Reference: "amanhã"
Normalized: {"relative": "tomorrow", "format": "day_name"}

Metric: Normalized F1
How often does predicted normalize to same value as reference?
```

### Annotation Guidelines for Evaluation

To build evaluation set, annotators should follow:

#### Temporal Spans
- **Include prepositions:** "às 15h" not just "15h" (but optional)
- **Exclude articles:** "a sexta" → "sexta", not include "a"
- **Include degree/ordinals:** "1º andar" inclusive

#### Participants
- **Include titles:** "Dr. Silva" as full span
- **Exclude initials only:** "JS" only if clearly mentioned (not inferred)
- **Include emails:** "name@company.com" as valid participant

#### Locations
- **Include building context:** "Bloco A, sala 203" as one span or two?
  - Recommendation: Extract separately if separated by comma
  - Together if compound: "bloco A sala 203"

#### Topics
- **Minimum 2 tokens:** "reunião" alone = too generic
- **Include modifiers:** "projeto de AI" not just "projeto"
- **Exclude pure discourse:** "eu", "nós", "é importante"

---

## 7. Evaluation Framework

### Example: Temporal Expression Evaluation

```python
def evaluate_temporal(predicted_spans, reference_spans, tolerance=2):
    """
    Evaluate temporal span extraction.
    
    Args:
        predicted_spans: List[ArgumentSpan] from extractor
        reference_spans: List[ArgumentSpan] from manual annotation
        tolerance: Allow span boundary to differ by N characters
    """
    def spans_match(pred, ref, tol=tolerance):
        # Exact match
        if pred.span_start == ref.span_start and pred.span_end == ref.span_end:
            return True
        # Partial with tolerance
        if (abs(pred.span_start - ref.span_start) <= tol and
            abs(pred.span_end - ref.span_end) <= tol):
            return True
        return False
    
    tp = sum(1 for p in predicted_spans if any(
        spans_match(p, r) for r in reference_spans
    ))
    fp = len(predicted_spans) - tp
    fn = len(reference_spans) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}
```

### Expected Baseline Performance

Based on similar NLP systems for Portuguese:

| Argument | Expected F1 | Reasoning |
|----------|-----------|-----------|
| Temporal | 0.75-0.85 | Regex patterns work well; ambiguity with informal expressions |
| Location | 0.70-0.80 | Good for standard patterns; context-dependent failures |
| Participants | 0.65-0.75 | spaCy domain shift; abbreviations challenge |
| Topic | 0.50-0.65 | Highly subjective; requires more context |

---

## 8. Evolution Path: From Regex to BERT Token Classification

### Current Baseline: Rule-Based

**Pros:**
- ✅ No labeled data needed
- ✅ Fast inference
- ✅ Interpretable (can explain each match)
- ✅ Works immediately for common patterns

**Cons:**
- ❌ Cannot handle context/negation ("Não às 15h")
- ❌ Limited to predefined patterns
- ❌ Difficult to combine signals (confidence unclear)
- ❌ No learning from errors

### Stage 1: Sequence Labeling (Intermediate)

Upgrade to `BIO` tagging on tokens:

```
Email: "Reunião segunda às 15h em sala 203"
Tokens: "Reunião"  "segunda"  "às"   "15h"  "em"   "sala"   "203"
Tags:   "O"        "B-TIME"   "I-TIME" "I-TIME" "O"  "B-LOC"  "I-LOC"
```

**Model:** Use CRF (Conditional Random Fields) or simple BiLSTM

**Advantages over regex:**
- Learn contextual patterns: "em [LOCATION]" is location marker
- Handle variations: "sala", "escritório", "bloco" learned as features
- Combine features: word form + suffix + POS tag

**Training data needed:** ~500-1000 annotated emails

**Tools:**
- `sklearn-crfsuite` for CRF
- `flair` or `transformers` for BiLSTM

### Stage 2: BERT Token Classification (Advanced)

Fine-tune pre-trained BERT on Portuguese meetings:

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    "neuralmind/bert-base-portuguese-cased",
    num_labels=7  # O, B-TIME, I-TIME, B-LOC, I-LOC, B-PER, I-PER, B-TOPIC, ...
)

# Fine-tune on annotated emails
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    args=training_args,
)
trainer.train()
```

**Advantages:**
- ~5-15% F1 improvement over CRF
- Learns semantic representations
- Better error recovery
- Can fine-tune on multi-task: classification + NER

**Data requirements:** ~2000-5000 annotated emails

**Training time:** 2-4 hours on single GPU

### Stage 3: Joint Learning

Combine argument extraction + intent classification:

```
Input: Email text
↓
BERT encoder (shared)
↓
├─→ Intent classifier head → agendamento_reuniao
└─→ Token classifier head → temporal/location/participant/topic spans
```

**Why better?**
- Intent can disambiguate topics
- Intent focuses argument extractor (e.g., agendamento → look for time)
- Shared representations learned together
- ~3-8% F1 improvement

### Roadmap

```
Month 1: Baseline (current)
         - Regex patterns, heuristics
         - Evaluate on 500 test emails

Month 2: Collect labels
         - Annotate 2000 emails with argument spans
         - Define schema (BIO tags)
         - Build evaluation set

Month 3: CRF model
         - Train CRF on 1500 emails
         - Compare: CRF vs Regex baseline
         - Error analysis

Month 4: BERT
         - Fine-tune Portuguese BERT
         - Data augmentation if needed
         - Hyperparameter tuning

Month 5: Joint learning
         - Design multi-task architecture
         - Train on intent + arguments
         - Final evaluation

Month 6: Deployment
         - API wrapper
         - Batch processing
         - A/B testing against baseline
```

---

## 9. Usage Example

```python
from preprocessing.argument_extraction import ArgumentExtractor

# Initialize extractor (first run downloads pt_core_news_sm)
extractor = ArgumentExtractor(model_name="pt_core_news_sm")

# Example email
email_body = """
Gostaria de agendar uma reunião com o Dr. Silva e Maria Costa para discutir 
o projeto de IA. Podemos fazer amanhã à tarde na sala 203? 

Caso não seja possível, próxima segunda de manhã também fica bem.

Obrigado,
João
"""

email_subject = "Reunião - Projeto de IA"
predicted_intent = "agendamento_reuniao"
trigger = "agendar"

# Extract arguments
result = extractor.extract_with_context(
    email_body=email_body,
    email_subject=email_subject,
    predicted_intent=predicted_intent,
    trigger=trigger,
)

# Result structure:
{
    'extracted_arguments': {
        'participants': [
            {'text': 'Dr. Silva', 'span_start': 30, 'span_end': 40, 'confidence': 0.85, 'extraction_method': 'ner_spacy'},
            {'text': 'Maria Costa', 'span_start': 46, 'span_end': 57, 'confidence': 0.82, 'extraction_method': 'ner_spacy'},
        ],
        'time_expressions': [
            {'text': 'amanhã', 'span_start': 110, 'span_end': 116, 'confidence': 0.90, 'extraction_method': 'regex_relative'},
            {'text': 'à tarde', 'span_start': 117, 'span_end': 124, 'confidence': 0.85, 'extraction_method': 'regex_informal'},
            {'text': 'próxima segunda', 'span_start': 167, 'span_end': 181, 'confidence': 0.90, 'extraction_method': 'regex_relative'},
            {'text': 'de manhã', 'span_start': 182, 'span_end': 190, 'confidence': 0.85, 'extraction_method': 'regex_informal'},
        ],
        'locations': [
            {'text': 'sala 203', 'span_start': 125, 'span_end': 133, 'confidence': 0.90, 'extraction_method': 'regex_location'},
        ],
        'topics': [
            {'text': 'projeto de IA', 'span_start': 73, 'span_end': 86, 'confidence': 0.88, 'extraction_method': 'spacy_noun_chunks'},
        ],
    },
    'metadata': {
        'email_subject': 'Reunião - Projeto de IA',
        'predicted_intent': 'agendamento_reuniao',
        'trigger': 'agendar',
        'extraction_timestamp': '2024-04-09T14:30:00.000000',
    }
}
```

---

## 10. Key Insights & Best Practices

1. **Exact spans matter:** For actionable information, preserve original text spans
2. **Confidence is uncertainty:** Always return confidence scores; downstream systems can make risk-vs-precision tradeoffs
3. **Method diversity:** Using multiple extraction methods (NER + regex + heuristics) is better than one approach
4. **Portuguese specificity:** Informal expressions are common; special handling needed
5. **Context is king:** Intent classification helps disambiguate arguments
6. **Incrementally improve:** Start with regex, move to ML as data accumulates
7. **Evaluate early:** Build evaluation set while building patterns; avoid over-fitting to examples

---

## References

- Linguistic patterns: [Portuguese syntax reference](https://en.wikipedia.org/wiki/Portuguese_language#Grammar)
- spaCy models: `python -m spacy download pt_core_news_sm`
- Temporal resolution: [Temporal Annotation Guidelines](https://timeml.github.io/)
- Token classification: [Hugging Face Task Guide](https://huggingface.co/tasks/token-classification)
