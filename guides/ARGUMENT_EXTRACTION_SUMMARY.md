# 📊 Argument Extraction Baseline - Complete Implementation Summary

## Executive Summary

I've created a **modular, production-ready baseline system** for extracting structured arguments from Portuguese meeting emails. The system extracts:

- **Participants** (using spaCy NER + email pattern matching)
- **Time Expressions** (Portuguese-specific temporal regex patterns)
- **Locations** (room/building pattern detection)
- **Topics** (noun chunks + keyword heuristics)

Each extraction returns **exact text spans** with confidence scores and extraction methods, enabling downstream systems to make risk-aware decisions.

---

## 1. System Architecture

```
Email Input
├─ Subject
├─ Body  
├─ Predicted Intent (from classifier)
└─ Trigger (optional, from extractor)
    ↓
    ├─→ TemporalExpressionExtractor (regex patterns)
    ├─→ LocationExtractor (regex + heuristics)
    ├─→ ParticipantExtractor (spaCy NER + patterns)
    └─→ TopicExtractor (noun chunks + keywords)
    ↓
ExtractedArguments
├─ participants: List[ArgumentSpan]
├─ time_expressions: List[ArgumentSpan]
├─ locations: List[ArgumentSpan]
└─ topics: List[ArgumentSpan]
```

Each `ArgumentSpan` contains:
```python
{
    "text": "exact text from email",
    "span_start": 123,           # Character position
    "span_end": 128,             # Character position
    "confidence": 0.85,          # 0-1 score
    "extraction_method": "ner_spacy"  # How extracted
}
```

---

## 2. Argument Extraction Strategy

### 2.1 Temporal Expression Extraction (Regex-Based)

**Why Regex?** Portuguese temporal expressions follow consistent linguistic patterns that don't require labeled training data.

**Handles 5 categories:**

| Category | Examples | Regex Patterns |
|----------|----------|---|
| **Specific Dates** | "5 de março", "15 de janeiro de 2024" | Date word patterns with day/month/year |
| **Weekdays** | "segunda-feira", "sexta", "seg.", "6ª" | Weekday names + abbreviations |
| **Relative Dates** | "amanhã", "próxima semana", "daqui a 3 dias" | Relative word patterns |
| **Times** | "15h", "15:00", "às 15 horas", "9 horas" | 24-hour format, with/without "às", "horas" |
| **Informal** | "de manhã", "depois de almoço", "de tarde" | Time-of-day expressions (critical for emails!) |

**Key Features:**
- ✅ Case-insensitive matching
- ✅ Handles variations: "seg.", "segunda-feira", "2ª"
- ✅ Informal Portuguese expressions common in emails
- ✅ Deduplication: longest-match strategy for overlaps
- ⚠️ Cannot handle negation ("Não amanhã") - post-processing needed

**Example:**
```
Email: "Reunião próxima segunda de tarde, entre 15-16h"
Output: ["próxima segunda", "de tarde", "15-16h"]
```

### 2.2 Location Extraction (Regex-Based)

**Categories:**

| Pattern | Examples | Regex |
|---------|----------|-------|
| **Rooms** | "sala 203", "escritório nº 5", "auditório A" | `sala/escritório/gabinete + number/letter` |
| **Floors** | "1º andar", "piso 2" | `digit + ordinal + andar/piso` |
| **Buildings** | "bloco A", "edifício 1" | `bloco/edifício + identifier` |
| **Named Spaces** | "laboratório", "biblioteca", "cantina" | Named building spaces |
| **Addresses** | "Rua da Prata, 50" | Street patterns with numbers |

**Confidence:** 0.85 (good pattern but context-dependent)

**Example:**
```
Email: "Reunião na sala 405, bloco B"
Output: ["sala 405", "bloco B"]
```

### 2.3 Participant Extraction (Multi-Source)

**Three extraction methods:**

1. **spaCy NER (Primary)**
   - Method: Identifies PERSON entities using Portuguese model
   - Confidence: 0.80
   - Strength: Handles titles ("Dr. Silva", "Eng. Costa")
   - Weakness: Domain shift (NER trained on news, not emails)

2. **Email Regex (Precise)**
   - Method: Regex pattern for `user@company.com`
   - Confidence: 0.95
   - Strength: Highly reliable pattern
   - Weakness: May not correspond to actual participant

3. **Pattern-Based Heuristics (Informal)**
   - Method: Context patterns like "com o João", "entre X e Y"
   - Confidence: 0.60
   - Strength: Catches informal references
   - Weakness: Many false positives

**Example:**
```
Email: "Reunião com o Dr. Silva, joão@company.com e Maria"
Output: ["Dr. Silva", "joão@company.com", "Maria"]
```

### 2.4 Topic Extraction (Heuristic-Based)

**Strategy:** Multi-approach ranking

1. **Noun Chunks (spaCy)**
   - Extract multi-word noun phrases
   - Filter stop words
   - Score by content density (non-stop-word ratio)
   - Confidence: 0.5-0.9

2. **Keyword Matching**
   - Predefined categories: projeto, orçamento, cronograma, qualidade, recursos, apresentação
   - Scan text for category keywords
   - Confidence: 0.70

3. **Intent Context**
   - Adjust weights based on predicted intent
   - Example: `agendamento_reuniao` → focus on meeting content, not scheduling

4. **Ranking & Limiting**
   - Rank by confidence (highest first)
   - Return top 5 topics only

**Example:**
```
Email: "Discutir cronograma do Q4 e recursos necessários"
Output: ["cronograma do Q4", "recursos", "Q4"]
Ranked by confidence (noun chunks first)
```

---

## 3. Handling Portuguese Specificity

### Informal Expressions

Portuguese emails frequently use informal temporal markers that challenge regex systems:

| Expression | Interpretation | Challenge |
|------------|---|---|
| "de manhã" | Morning (implies ~8AM-12PM) | Context-dependent time range |
| "depois de almoço" | After lunch (~1-5PM) | Portuguese-specific |
| "de tarde" | Afternoon | Vague time range |
| "à noite" | Evening | Vague boundaries |
| "de fim de semana" | Weekend | Relative to current date |
| "em breve" | Soon (undefined) | No specific time |

**Current Approach:** Extract as-is with confidence 0.85 (pattern matches but semantics unclear)

**Future:** Map to time ranges using heuristics or learning

### Multiple Expressions in Sequence

Email: "Reunião próxima segunda-feira, de manhã, às 10h"
- Extracts: ["próxima segunda-feira", "de manhã", "às 10h"]
- Deduplication logic: Longest-match keeps non-overlapping spans
- Use case: Combine for richer temporal understanding

---

## 4. Known Failure Cases & Mitigations

### Temporal

| Failure | Example | Cause | Mitigation |
|---------|---------|-------|-----------|
| Implicit dates | "Confirmo para quinta" | No day reference → needs calendar context | Store extraction date; convert to absolute |
| Negation | "Não às 15h, sim 16h" | Regex catches both | Add negation pre-filtering |
| Duration confusion | "reunião de 1 hora" | Captures "1 hora" as time | Filter patterns following "de" (duration marker) |
| Informal only | "Na hora do almoço" | Too vague | Model as time-range "12:00-13:00" with low confidence |

### Locations

| Failure | Example | Cause | Mitigation |
|---------|---------|-------|-----------|
| Generic refs | "na sala de espera" | "sala" matches but wrong type | Intent filter: agendamento → assume meeting room |
| Multiple locs | "Sala 203, bloco A" | May be separate or compound | Extract separately; merge if syntactically adjacent |
| Uninformative | "o local" | Just pronoun | Filter generic references |

### Participants

| Failure | Example | Cause | Mitigation |
|---------|---------|-------|-----------|
| Signature names | 40 names in footer | Unrelated to meeting | Pre-process email to remove signature (look for "--") |
| Abbreviations | "JS e MPS" | No context for expansion | Add company directory lookup |
| Title confusion | "Aqui está a proposta, o Dr." | May be incomplete | Reduce confidence for edge cases |

### Topics

| Failure | Example | Cause | Mitigation |
|---------|---------|-------|-----------|
| Generic topics | "reunião", "discussão" | Too common | Add to stop words |
| Missing names | "Novo projeto X" | X undefined | Keep as topic; note low confidence |
| Email chains | Forwarded message | Multiple meeting topics mixed | Pre-process: split on "---" or "From:" |

---

## 5. Evaluation Metrics for Span Extraction

### Exact Match (Strict)

```python
Predicted span: "15h" at character 10-13
Reference:     "às 15h" at character 8-13
Result: NO MATCH (different start position)

F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### Partial Match (Relaxed)

Allow tolerance of ±2 characters:
```python
Predicted: "15h" [10-13]
Reference: "às 15h" [8-13]
Same semantic value → MATCH
```

### Category-Specific Metrics

**Temporal:** Did system normalize to same relative date?
```
Email: "amanhã às 15h"
System: {relative: "tomorrow", time: "15:00"}
Reference: {relative: "tomorrow", time: "15:00"}
→ F1 on normalized values
```

**Participant:** Email vs full name accuracy
```
Email: "js@company.com"
System: "js@company.com"  
Reference: "João Silva"
→ Partial credit for email, full credit if name resolved
```

### Expected Baseline Performance

| Argument | F1 Score | Reasoning |
|----------|----------|-----------|
| Temporal | 0.75-0.85 | Regex patterns work well; informal expressions challenging |
| Location | 0.70-0.80 | Good for standard rooms; context-dependent failures |
| Participants | 0.65-0.75 | spaCy domain shift; abbreviations problematic |
| Topics | 0.50-0.65 | Highly subjective; needs more context/training |

---

## 6. Evolution Path: Regex → BERT Token Classification

### Stage 0: Current Baseline (Rule-Based)

**Pros:**
- ✅ Works immediately, no labeled data
- ✅ Fast inference
- ✅ Interpretable (can explain each match)
- ✅ Handles Portuguese informal expressions well

**Cons:**
- ❌ Cannot understand context/negation
- ❌ Limited to predefined patterns
- ❌ Cannot combine signals intelligently

### Stage 1: CRF Sequence Labeling (3 months)

Use conditional random fields with token features:

```
Email: "Reunião segunda às 15h em sala 203"
Tokens: [Reunião] [segunda] [às] [15h] [em] [sala] [203]
BIO:    [O]      [B-TIME] [I-TIME] [I-TIME] [O] [B-LOC] [I-LOC]

Features: word form + POS tag + suffix + capitalization
Model: CRF learns patterns from labeled data
```

**Requirements:**
- ~500-1000 annotated emails
- Training time: Hours
- F1 improvement: ~5-10%

**Tools:** `sklearn-crfsuite`, `flair`

### Stage 2: BERT Token Classification (6 months)

Fine-tune Portuguese BERT (neuralmind/bert-base-portuguese-cased):

```python
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    "neuralmind/bert-base-portuguese-cased",
    num_labels=7  # O, B-TIME, I-TIME, B-LOC, I-LOC, B-PER, I-PER, ...
)
```

**Advantages:**
- ~15-20% F1 improvement over regex
- Better context understanding
- Learns semantic representations
- Error recovery

**Requirements:**
- ~2000-5000 annotated emails
- GPU training: 2-4 hours
- Inference: 100-200 emails/sec on CPU

**When to transition?**
- After building evaluation set (~500 emails)
- When baseline fails on >20% of cases
- When you have labeled data

### Transition Strategy

```
Month 1-2: Current (regex baseline)
          - Evaluate on 500 test emails
          - Collect failure examples
          - Build annotation guidelines

Month 3-4: CRF model
          - Annotate 1500 emails
          - Train CRF on 80/20 split
          - Compare: Regex vs CRF vs Hybrid

Month 5-6: BERT transfer learning
          - Fine-tune Portuguese BERT
          - Multi-task: intent + arguments
          - Production evaluation

Month 7+: Deployment & refinement
          - A/B test against baseline
          - Collect hard examples
          - Continue learning
```

---

## 7. How to Evaluate Span Extraction Quality

### Build Evaluation Set

```python
# Annotate 500 emails with argument spans
annotation_schema = {
    "email_id": "string",
    "email_body": "string",
    "arguments": {
        "participants": [
            {"text": "João Silva", "span_start": 30, "span_end": 41}
        ],
        "times": [
            {"text": "amanhã", "span_start": 50, "span_end": 56}
        ],
        "locations": [
            {"text": "sala 203", "span_start": 80, "span_end": 88}
        ],
        "topics": [
            {"text": "projeto de IA", "span_start": 100, "span_end": 112}
        ]
    }
}
```

### Compute Metrics

```python
def evaluate_extraction(predicted_spans, reference_spans):
    """Compute P, R, F1 for span extraction."""
    
    # Exact match with ±2 character tolerance
    def spans_match(pred, ref, tol=2):
        return (abs(pred['span_start'] - ref['span_start']) <= tol and
                abs(pred['span_end'] - ref['span_end']) <= tol)
    
    tp = sum(1 for p in predicted_spans 
             if any(spans_match(p, r) for r in reference_spans))
    fp = len(predicted_spans) - tp
    fn = len(reference_spans) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}
```

### Error Analysis

```python
# Categorize failures
false_positives = [...]  # System extracted but shouldn't
false_negatives = [...]  # System missed but should extract
incorrect_spans = [...]  # Right text, wrong boundaries

# By category
temporal_errors = [e for e in all_errors if e['type'] == 'temporal']
participant_errors = [e for e in all_errors if e['type'] == 'participant']

# Pattern analysis
for error in temporal_errors:
    if "de tarde" in error['text']:
        print(f"Informal time issue: {error}")
    elif error['span_start'] != reference_span_start:
        print(f"Boundary issue: {error}")
```

---

## 8. File Structure & Usage

### New Files Created

```
preprocessing/
├── argument_extraction.py              (Main module, 600+ lines)
└── email_pipeline_enhanced.py          (Integration with pipeline)

test_argument_extraction.py             (Test suite with examples)

ARGUMENT_EXTRACTION_GUIDE.md            (Detailed strategy & explanation)
ARGUMENT_EXTRACTION_QUICKSTART.md       (API reference & quick start)
ARGUMENT_EXTRACTION_SUMMARY.md          (This file)
```

### Quick Start

```python
from preprocessing.argument_extraction import ArgumentExtractor

# Initialize
extractor = ArgumentExtractor(model_name="pt_core_news_sm")

# Extract
result = extractor.extract_with_context(
    email_body="Reunião amanhã às 15h na sala 203 com João",
    email_subject="Subject",
    predicted_intent="agendamento_reuniao",
    include_confidence=True,
)

# Access
print(result["extracted_arguments"]["participants"])
# [{'text': 'João', 'confidence': 0.85, 'extraction_method': 'ner_spacy', ...}]
```

### Integration with Existing Pipeline

```python
from preprocessing.email_pipeline_enhanced import EnhancedEmailAnalysisPipeline

# Creates structured output combining:
# - Intent classification
# - Trigger extraction  
# - Argument extraction

pipeline = EnhancedEmailAnalysisPipeline()
analysis = pipeline.analyze(
    email_subject="...",
    email_body="...",
    predicted_intent="...",
    intent_confidence=0.92,
)

print(analysis.actionable_summary())
# 📅 Agendamento Reuniao
# 👥 With: João Silva
# ⏰ When: amanhã, às 15h
# 📍 Where: sala 203
# 📎 About: projeto
```

---

## 9. Key Design Decisions

### 1. Exact Text Spans (Not Normalized)

**Decision:** Return extracted text exactly as-is in email, with span positions

**Rationale:**
- Downstream systems need original text for confirmation/UI
- Normalization (e.g., "amanhã" → "2024-04-10") requires calendar context
- Preserves user intent without lossy transformation

**Trade-off:** Parsing system must handle normalization separately

### 2. Multiple Extraction Methods

**Decision:** Use spaCy NER + Regex + Heuristics, report method for each span

**Rationale:**
- Different methods have different strengths (NER for names, Regex for dates)
- Confidence varies by method (Email regex > spaCy NER > heuristics)
- Downstream can implement filtering by method if needed

**Trade-off:** More complex than single method, but more robust

### 3. No Multi-Task Learning (Yet)

**Decision:** Separate extraction modules, not joint model

**Rationale:**
- Baseline should be rule-based (no labeled data needed)
- Modules are reusable and testable independently
- Evolution path clear: Regex → CRF → BERT

**Trade-off:** Cannot share representations; upgrade to joint learning later

### 4. Regex-Primary for Temporal

**Decision:** Regex patterns before any ML

**Rationale:**
- Portuguese temporal expressions follow consistent grammar
- ML requires labeled data; regex works immediately
- Can evaluate regex performance on evaluation set
- Easy to add patterns incrementally

**Trade-off:** Cannot handle context or novel patterns; but good baseline

---

## 10. Limitations & Future Work

### Current Limitations

1. **No Negation Handling** - "Não amanhã" extracts "amanhã"
   - Fix: Pre-filter spans with "não" in preceding 15 chars

2. **No Disambiguation** - "sala" could be meeting room or waiting room
   - Fix: Use intent context (agendamento → meeting room prioritized)

3. **No Abbreviation Expansion** - "JS" not recognized as "João Silva"
   - Fix: Build company directory lookup

4. **No Signature Removal** - Extracts participant names from email footer
   - Fix: Pre-process to remove signature (look for standalone "--")

5. **Informal Time Ranges Vague** - "de tarde" doesn't map to specific times
   - Fix: Heuristically map to ranges (12:00-18:00) or train on annotations

### Roadmap

| Phase | Timeline | Work |
|-------|----------|------|
| **Current** | Now | Regex baseline, evaluation set collection |
| **Enhancement** | 1-2 weeks | Negation filtering, signature removal |
| **Evaluation** | 1 month | Annotate 500 emails, error analysis |
| **CRF Model** | 2-3 months | Transition to sequence labeling |
| **BERT** | 4-6 months | Fine-tune transformer, multi-task learning |
| **Production** | 6+ months | Deployment, online learning, A/B testing |

---

## 11. Summary: Why This Baseline?

### What Makes It a Good Baseline

✅ **No Labeled Data Required** - Regex + heuristics work out of the box  
✅ **Interpretable** - Can explain each extraction (exact regex match)  
✅ **Fast** - Inference in milliseconds per email  
✅ **Modular** - Each argument type independently extracted  
✅ **Explainable** - Spans, confidence, and method included  
✅ **Buildable** - Clear evolution path to ML systems  

### Realistic Expectations

- **Temporal:** 75-85% F1 (Regex patterns good, informal expressions hard)
- **Location:** 70-80% F1 (Standard rooms well-defined, ambiguity remains)
- **Participants:** 65-75% F1 (spaCy domain shift, abbreviations challenging)
- **Topics:** 50-65% F1 (Highly subjective, needs training data)

### When to Transition to ML

Move to CRF/BERT when:
1. Baseline fails on >20% of emails
2. Error analysis shows consistent patterns
3. You've collected 1000+ annotated examples
4. Current performance insufficient for use case

---

## 12. Running the Code

### Test Examples

```bash
# Run test suite (requires spaCy model)
python test_argument_extraction.py

# Partial output shows:
# - Temporal pattern variants
# - Location recognition
# - Error cases explanation
# - Full pipeline integration
```

### Integration Example

```bash
# Run enhanced pipeline example
python preprocessing/email_pipeline_enhanced.py

# Shows complete flow:
# - Intent prediction
# - Trigger extraction
# - Argument extraction
# - Actionable summary output
```

### Component Testing

```python
# Test individual components
from preprocessing.argument_extraction import TemporalExpressionExtractor

temporal = TemporalExpressionExtractor()
spans = temporal.extract("Reunião amanhã às 15h")
print([s.text for s in spans])
# ['amanhã', 'às 15h']
```

---

## References & Resources

### Documentation
- **ARGUMENT_EXTRACTION_GUIDE.md** - Complete strategy & failure analysis
- **ARGUMENT_EXTRACTION_QUICKSTART.md** - API reference & usage examples
- **test_argument_extraction.py** - Test cases & demonstrations

### Portuguese NLP
- spaCy Model: `pt_core_news_sm-3.8.0`
- Documentation: https://spacy.io/models/pt

### For ML Evolution
- Named Entity Recognition: https://huggingface.co/tasks/token-classification
- Portuguese BERT: https://huggingface.co/neuralmind/bert-base-portuguese-cased
- Temporal Resolution: https://timeml.github.io/

---

## Support & Troubleshooting

**Problem:** "Model 'pt_core_news_sm' not found"
```bash
python -m spacy download pt_core_news_sm
```

**Problem:** Low participant extraction quality
- Likely: spaCy domain shift + abbreviations
- Check: Email patterns, try signature removal pre-processing
- Solution: Collect labeled data for CRF/BERT

**Problem:** Temporal expression ambiguity
- Likely: Informal expressions ("de tarde" = 12-18h?)
- Check: Error analysis for pattern frequency
- Solution: Add context/calendar integration

---

**Implementation by:** NLP Engineer  
**Project:** Automatic Event & Temporal Expression Recognition in Portuguese Emails  
**Date:** April 2024  
**Status:** Production-ready baseline, ML evolution path clear
