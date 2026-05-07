# System Architecture & Data Flow Diagrams

## 1. Overall Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EMAIL INPUT                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Subject: "Reunião - Projeto Q2"                         │   │
│  │ Body: "Gostaria de agendar com João, amanhã às 15h...   │   │
│  │        na sala 203 para discutir cronograma"            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 PIPELINE STAGE 1: INTENT CLASSIFICATION          │
│ (not part of this module, from models/predict_intent.py)        │
│                                                                  │
│  Input: "Gostaria de agendar..."                               │
│  ↓                                                              │
│  [TF-IDF Vectorization] → [Logistic Regression]               │
│  ↓                                                              │
│  Output: predicted_intent = "agendamento_reuniao"             │
│          confidence = 0.92                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  PIPELINE STAGE 2: TRIGGER EXTRACTION            │
│ (from preprocessing/trigger_extraction.py)                      │
│                                                                  │
│  Input: intent="agendamento_reuniao", text="Gostaria de..."    │
│  ↓                                                              │
│  [Pattern Matching] → [Lemmatization] → [Context Analysis]    │
│  ↓                                                              │
│  Output: trigger = "agendar"                                  │
│          method = "exact_match"                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│             PIPELINE STAGE 3: ARGUMENT EXTRACTION (NEW!)        │
│                    [THIS IMPLEMENTATION]                         │
│                                                                  │
│  Input: email_body, subject, predicted_intent, trigger         │
│  ↓                                                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ ArgumentExtractor (Main Orchestrator)                   │  │
│  │                                                          │  │
│  │ 1. Load spaCy model (pt_core_news_sm)                  │  │
│  │ 2. Initialize component extractors:                     │  │
│  │    - TemporalExpressionExtractor                       │  │
│  │    - LocationExtractor                                 │  │
│  │    - ParticipantExtractor                              │  │
│  │    - TopicExtractor                                    │  │
│  │                                                          │  │
│  │ 3. Run extraction on email body                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Output:                                                        │
│    participants: [ArgumentSpan, ArgumentSpan, ...]            │
│    time_expressions: [ArgumentSpan, ...]                      │
│    locations: [ArgumentSpan, ...]                             │
│    topics: [ArgumentSpan, ...]                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STRUCTURED OUTPUT                            │
│                                                                  │
│  ExtractedArguments {                                          │
│    participants: [                                             │
│      {text: "João", confidence: 0.85, method: "ner_spacy"}    │
│    ],                                                          │
│    time_expressions: [                                         │
│      {text: "amanhã", confidence: 0.90, method: "regex"},     │
│      {text: "às 15h", confidence: 0.90, method: "regex"}      │
│    ],                                                          │
│    locations: [                                               │
│      {text: "sala 203", confidence: 0.90, method: "regex"}    │
│    ],                                                          │
│    topics: [                                                  │
│      {text: "cronograma", confidence: 0.75, method: "kw"}     │
│    ]                                                          │
│  }                                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Architecture

### Temporal Expression Extractor

```
TemporalExpressionExtractor
│
├─ Regex Patterns (Compiled on init)
│  ├─ WEEKDAYS → r'\b(?:segunda|seg)\.?\s*-?\s*(?:feira)?\b'
│  ├─ RELATIVE_PATTERNS → ['amanhã', 'próxima semana', ...]
│  ├─ INFORMAL_TIME → ['de manhã', 'depois de almoço', ...]
│  ├─ TIME_PATTERNS → [r'\b(?:às|as|a)\s+(?:\d{1,2}):?...']
│  └─ DATE_PATTERNS → [r'\b\d{1,2}\s+de\s+(?:janeiro|...']
│
├─ extract(text) → List[ArgumentSpan]
│  ├─ Apply all regex patterns to text
│  ├─ Collect matches with metadata
│  ├─ Deduplicate overlaps (longest-match strategy)
│  └─ Sort by appearance in text
│
└─ _deduplicate_spans() → List[ArgumentSpan]
   ├─ Sort by start position + length
   ├─ Remove overlapping spans
   └─ Keep longest non-overlapping matches


Example Execution:
─────────────────
Input text: "Reunião próxima segunda de tarde às 15h"

Step 1: Apply patterns
  WEEKDAYS: "próxima segunda" at [9-24] ✓
  INFORMAL_TIME: "de tarde" at [25-33] ✓
  TIME_PATTERNS: "às 15h" at [34-40] ✓

Step 2: Collect results
  [
    ArgumentSpan("próxima segunda", 9, 24, 0.90, "regex_weekday"),
    ArgumentSpan("de tarde", 25, 33, 0.85, "regex_informal"),
    ArgumentSpan("às 15h", 34, 40, 0.90, "regex_time"),
  ]

Step 3: Deduplicate (no overlaps)
  Result: All 3 spans returned (no overlaps to remove)

Output: 3 ArgumentSpan objects
```

---

### Location Extractor

```
LocationExtractor
│
├─ Regex Patterns (Compiled on init)
│  ├─ Room patterns: r'\b(?:sala|escritório|auditório)\s+(?:de\s+)?(?:n°|nº|#)?(\d+|[A-Z])\b'
│  ├─ Floor patterns: r'\b(?:\d+)[°º]\s*(?:andar|piso)\b'
│  ├─ Named spaces: r'\b(?:auditório|laboratório|biblioteca|...)\b'
│  ├─ Building refs: r'\b(?:bloco|edifício)\s+[A-Z0-9]+\b'
│  └─ Address patterns: r'\b(?:rua|avenida|av|...)\s+[A-Z][a-z\s]+\b'
│
├─ extract(text) → List[ArgumentSpan]
│  ├─ Apply all regex patterns
│  ├─ Return matches with confidence 0.85
│  └─ Deduplicate overlaps
│
└─ Same deduplication as TemporalExtractor


Example:
───────
Input: "Reunião no 1º andar, sala 304, bloco B"

Matches:
  "1º andar" [12-20] floor pattern
  "sala 304" [23-31] room pattern
  "bloco B" [34-41] building pattern

No overlaps → All 3 returned
```

---

### Participant Extractor

```
ParticipantExtractor(nlp_model)
│
├─ init: Store loaded spaCy model
│  └─ Model has NER component trained on Portuguese news
│
├─ extract(text, doc=None) → List[ArgumentSpan]
│  │
│  ├─ Method 1: spaCy NER
│  │  └─ If doc not provided:
│  │     ├─ doc = nlp(text)  [spaCy processing]
│  │     └─ Extract all entities with label_ == "PER"
│  │
│  ├─ Method 2: Email Regex
│  │  └─ r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
│  │
│  ├─ Method 3: Pattern-Based Heuristics
│  │  ├─ r'(?:com|entre|por)\s+(?:o\s+|a\s+)([A-Z][a-záéíóúàâêõç\s]+?)'
│  │  └─ Context patterns: "com João", "entre X e Y", etc.
│  │
│  └─ Combine all spans
│
└─ _deduplicate_spans() → List[ArgumentSpan]
   └─ Keep highest confidence version of each unique name (case-insensitive)


Example:
───────
Input: "Reunião com Dr. Silva e joão@company.com"

spaCy NER: "Dr. Silva" [15-24] confidence=0.85
Email regex: "joão@company.com" [29-44] confidence=0.95
Patterns: (no matches)

Deduplication:
  Map by name (lowercase):
  "dr. silva" → best version from NER
  "joão@company.com" → best version from email regex

Output:
  [
    ArgumentSpan("Dr. Silva", 15, 24, 0.85, "ner_spacy"),
    ArgumentSpan("joão@company.com", 29, 44, 0.95, "regex_email"),
  ]
```

---

### Topic Extractor

```
TopicExtractor(nlp_model)
│
├─ STOP_WORDS: {'de', 'do', 'reunião', 'email', ...}
├─ TOPIC_KEYWORDS: {
│    'projeto': ['projeto', 'desenvolvimento', ...],
│    'orçamento': ['orçamento', 'preço', ...],
│    'cronograma': ['cronograma', 'timeline', ...],
│    ...
│  }
│
├─ extract(text, subject="", intent="") → List[ArgumentSpan]
│  │
│  ├─ Combine text + subject for analysis
│  │
│  ├─ Strategy 1: Noun Chunks
│  │  ├─ doc = nlp(combined_text)
│  │  └─ for chunk in doc.noun_chunks:
│  │     ├─ Filter stop words
│  │     ├─ Score by content density
│  │     └─ Create ArgumentSpan (confidence: 0.5-0.9)
│  │
│  ├─ Strategy 2: Keyword Matching
│  │  ├─ for each topic_category in TOPIC_KEYWORDS:
│  │  │  ├─ for keyword in category.keywords:
│  │  │  │  └─ Find matches in text
│  │  │  └─ Create ArgumentSpan (confidence: 0.70)
│  │  └─ Add category as topic
│  │
│  ├─ Strategy 3: Intent Context
│  │  └─ Adjust confidence based on predicted intent
│  │
│  └─ Deduplicate & rank by confidence
│
└─ Return top 5 topics ranked by confidence


Example:
───────
Input: 
  text="Discutir cronograma do Q4 e alocação de recursos"
  subject="Reunião de planejamento"
  intent="reuniao_confirmada"

Strategy 1: Noun Chunks
  "cronograma do Q4" [9-24] → 0.70 confidence
  "alocação de recursos" [29-49] → 0.75 confidence

Strategy 2: Keywords
  "cronograma" matches → topic = "cronograma" (0.70)
  "recursos" matches → topic = "recursos" (0.70)

Deduplication (by text, lowercase):
  "cronograma do Q4" → 0.70 (noun chunks)
  "alocação de recursos" → 0.75 (noun chunks)
  "cronograma" → 0.70 (keywords) → DUPLICATE, keep longer
  "recursos" → 0.70 (keywords) → DUPLICATE, keep longer

Ranking (by confidence):
  1. "alocação de recursos" (0.75)
  2. "cronograma do Q4" (0.70)

Output: Top 5 (only 2 available)
```

---

## 3. Argument Span Data Structure

```python
@dataclass
class ArgumentSpan:
    """One extracted argument with metadata."""
    
    # Content
    text: str                    # "João Silva"
    span_start: int             # 15 (character offset)
    span_end: int               # 26
    
    # Quality
    confidence: float = 1.0     # 0.85 (0-1, higher=better)
    extraction_method: str = "" # "ner_spacy", "regex", "heuristic"
    
    # Methods
    def to_dict() -> Dict: ...  # Serialize to JSON
    
    
ExtractedArguments:
    participants: List[ArgumentSpan]
    time_expressions: List[ArgumentSpan]
    locations: List[ArgumentSpan]
    topics: List[ArgumentSpan]


Example JSON:
{
  "text": "João Silva",
  "span_start": 15,
  "span_end": 26,
  "confidence": 0.85,
  "extraction_method": "ner_spacy"
}
```

---

## 4. Information Flow: Email → Arguments

```
Email:
"Gostaria de agendar uma reunião com João Silva amanhã às 15h 
 na sala 203 para discutir o cronograma."

character indices:
0         10        20        30        40        50        60        70        80        90
Gostaria d[e agendar uma reunião com João S]ilva amanhã às 15h [na sala 203] para discutir [o cronograma]
          ^[42:57]^                        ^[46:57]^               ^[73:83]^              ^[93:105]^


Component Extraction:
────────────────────

1. TEMPORAL EXTRACTOR
   Pattern matching on entire text
   ├─ RELATIVE: matches "amanhã" at (0, 66, 72)
   ├─ INFORMAL: matches "às 15h" at (73, 80, 86)  ✗ No match (not informal)
   ├─ TIME: matches "às 15h" at (74, 80, 86)
   └─ Deduplicate: "amanhã" [66-72], "às 15h" [74-80]


2. LOCATION EXTRACTOR
   Pattern matching on entire text
   ├─ Room pattern: matches "sala 203" at (87, 91, 99)
   └─ Result: "sala 203" [87-99]


3. PARTICIPANT EXTRACTOR
   a) spaCy NER:
      └─ Identifies PER: "João Silva" [46-57]
   
   b) Email regex:
      └─ No matches
   
   c) Patterns:
      └─ "com João Silva" matches pattern "com + name"
         Extracts "João Silva" [46-57]
   
   Deduplicate:
   └─ "joão silva" → keep NER version (confidence 0.85)


4. TOPIC EXTRACTOR
   a) Noun chunks:
      ├─ "reunião" [24-32] → stop word, skip
      ├─ "cronograma" [100-110] → keep (0.70)
   
   b) Keywords:
      ├─ "cronograma" matches → "cronograma" (0.70)
   
   Result: "cronograma" [100-110]


Final Output:
─────────────
ExtractedArguments {
  participants: [
    ArgumentSpan(
      text="João Silva",
      span_start=46,
      span_end=57,
      confidence=0.85,
      extraction_method="ner_spacy"
    )
  ],
  
  time_expressions: [
    ArgumentSpan(
      text="amanhã",
      span_start=66,
      span_end=72,
      confidence=0.90,
      extraction_method="regex_relative"
    ),
    ArgumentSpan(
      text="às 15h",
      span_start=74,
      span_end=80,
      confidence=0.90,
      extraction_method="regex_time"
    )
  ],
  
  locations: [
    ArgumentSpan(
      text="sala 203",
      span_start=87,
      span_end=95,
      confidence=0.90,
      extraction_method="regex_location"
    )
  ],
  
  topics: [
    ArgumentSpan(
      text="cronograma",
      span_start=100,
      span_end=110,
      confidence=0.70,
      extraction_method="keyword_heuristic"
    )
  ]
}
```

---

## 5. Confidence Score Sources

```
┌─────────────────────────────────────────────────────────────┐
│  EXTRACTION METHOD CONFIDENCE MAPPING                       │
└─────────────────────────────────────────────────────────────┘

TEMPORAL EXPRESSIONS:
  regex_date:       0.90  (Precise patterns, very reliable)
  regex_weekday:    0.90  (Weekdays well-defined)
  regex_relative:   0.90  (Relative expressions clear)
  regex_time:       0.90  (Time patterns precise)
  regex_informal:   0.85  (Informal times less precise)

LOCATIONS:
  regex_location:   0.85-0.90  (Pattern-based, context-dependent)

PARTICIPANTS:
  ner_spacy:        0.80  (NER trained on news, domain drift)
  regex_email:      0.95  (Email pattern very precise)
  regex_heuristic:  0.60  (Pattern-based, many false positives)

TOPICS:
  spacy_noun_chunks: 0.50-0.90  (Content density weighted)
  keyword_heuristic: 0.70  (Keyword presence)


Decision Logic for Users:
  confidence > 0.90: High confidence, use directly
  0.70 < confidence < 0.90: Medium confidence, review/validate
  confidence < 0.70: Low confidence, high error rate
```

---

## 6. Processing Pipeline with Profiling

```
Input Email (Example: 500 characters)
  │
  ├─→ [Init: 50ms] Load spaCy model (cached after first run)
  │
  ├─→ ArgumentExtractor.extract(email_body)
  │   │
  │   ├─→ [1ms] TemporalExpressionExtractor.extract()
  │   │   ├─ Regex compilation: 0.1ms
  │   │   ├─ Pattern matching: 0.5ms
  │   │   ├─ Deduplication: 0.2ms
  │   │   └─ Result: 3-5 temporal spans
  │   │
  │   ├─→ [2ms] LocationExtractor.extract()
  │   │   ├─ Regex matching: 1ms
  │   │   ├─ Deduplication: 0.5ms
  │   │   └─ Result: 1-3 location spans
  │   │
  │   ├─→ [15ms] ParticipantExtractor.extract()
  │   │   ├─ spaCy NER: 12ms (most expensive!)
  │   │   ├─ Email regex: 1ms
  │   │   ├─ Pattern heuristics: 1ms
  │   │   ├─ Deduplication: 0.5ms
  │   │   └─ Result: 1-5 participant spans
  │   │
  │   ├─→ [15ms] TopicExtractor.extract()
  │   │   ├─ spaCy noun chunks: 12ms
  │   │   ├─ Keyword matching: 1.5ms
  │   │   ├─ Ranking: 1ms
  │   │   └─ Result: 1-5 topic spans
  │   │
  │   └─→ Aggregate results
  │
  └─→ Total: ~35-40ms per email
      (After first spaCy load: ~20-25ms)

Batch Processing (1000 emails):
  ├─ Init once: 50ms
  ├─ Per email: 25ms × 1000 = 25,000ms
  └─ Total: ~25 seconds (1 email every 25ms)

Optimization:
  ├─ Reuse extractor instance (✓ implemented)
  ├─ Batch processing with threading (future)
  └─ Component-only extraction if needed (supported)
```

---

## 7. Error Cascade Prevention

```
Input Email
  │
  ├─ Quality Check
  │  ├─ if len(text) < 10: SKIP (too short)
  │  ├─ if text has HTML tags: CLEAN (pre-processing)
  │  └─ Continue
  │
  ├─ Temporal Extraction
  │  ├─ Regex matches found? Continue
  │  └─ if no matches: Add empty list (no error)
  │
  ├─ Location Extraction  
  │  ├─ Regex matches found? Continue
  │  └─ if no matches: Add empty list (no error)
  │
  ├─ Participant Extraction
  │  ├─ spaCy NER runs: Always succeeds
  │  ├─ Email regex runs: Always succeeds
  │  ├─ if all methods fail: Add empty list
  │  └─ Continue
  │
  ├─ Topic Extraction
  │  ├─ spaCy processing: Always succeeds
  │  ├─ if no topics found: Add empty list
  │  └─ Continue
  │
  └─ Output ExtractedArguments (never fails)
     ├─ participants: [...] (possibly empty)
     ├─ time_expressions: [...] (possibly empty)
     ├─ locations: [...] (possibly empty)
     └─ topics: [...] (possibly empty)


Error Handling Policy:
  • No extraction should raise exception
  • Missing data = empty list (not None or error)
  • Always return ExtractedArguments object
  • Document extraction method for each span
  • Return confidence scores for downstream decisions
```

---

## 8. Evaluation Metrics Visualization

```
Test Email:
"Reunião amanhã com João na sala 203 para discutir cronograma"

Reference (Human Annotation):
  participants: ["João"]
  times: ["amanhã"]
  locations: ["sala 203"]
  topics: ["cronograma"]

System Output:
  participants: ["João"]
  times: ["amanhã", "de dia"]  ← EXTRA (false positive)
  locations: ["sala 203"]
  topics: ["cronograma", "discussão"]  ← EXTRA (false positive)

Metrics Calculation:
  
  Participants:
    TP=1, FP=0, FN=0 → P=100%, R=100%, F1=100%
  
  Times:
    TP=1, FP=1, FN=0
    P = 1/(1+1) = 50%
    R = 1/(1+0) = 100%
    F1 = 2*0.5*1/(0.5+1) = 66.7%
  
  Locations:
    TP=1, FP=0, FN=0 → P=100%, R=100%, F1=100%
  
  Topics:
    TP=1, FP=1, FN=0
    P = 1/(1+1) = 50%
    R = 1/(1+0) = 100%
    F1 = 2*0.5*1/(0.5+1) = 66.7%
  
  Macro-average F1 = (100% + 66.7% + 100% + 66.7%) / 4 = 83.4%


Error Types:
  False Positive: System extracted but shouldn't
    - "de dia" detected as time when not intended
    - "discussão" as topic vs generic discourse
  
  False Negative: System missed but should extract
    - Would occur if participant not recognized by NER
  
  Partial Match: Right content, wrong span
    - System: "João S" vs Reference: "João Silva"
```

---

This architecture ensures:
1. **Modularity** - Each component independent
2. **Robustness** - No cascading failures
3. **Interpretability** - Method & confidence visible
4. **Performance** - ~25ms per email
5. **Debuggability** - Clear data flow
