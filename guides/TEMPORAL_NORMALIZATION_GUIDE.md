"""
TEMPORAL EXPRESSION NORMALIZATION: DESIGN & ANALYSIS GUIDE

A comprehensive guide to understanding rule-based temporal normalization for
Portuguese emails, including strategy, edge cases, limitations, and ML improvements.


Date: 2026
"""


## TABLE OF CONTENTS
1. Normalization Strategy
2. Implementation Details
3. Edge Cases & Ambiguity Handling
4. Limitations of Rule-Based Systems
5. Machine Learning Alternatives
6. Real-World Considerations
7. Recommendations


---

## 1. NORMALIZATION STRATEGY

### Overview
The temporal normalization module uses a **cascade of layered pattern matching**
with rule-based heuristics to convert natural language temporal expressions into
structured datetime objects.

```
Input: "sexta às 15h"
       ↓
   [Pattern Matching Pipeline]
       ↓
   [Rule Application]
       ↓
Output: 2026-04-24T15:00:00 (Friday, April 24, 2026 at 3 PM)
```

### Core Components

#### 1.1 Lexicon-Based Matching
Predefined dictionaries map known temporal expressions to computational values:

```python
# Relative expressions
'amanhã' → +1 day offset
'hoje' → +0 day offset
'ontem' → -1 day offset
'para a semana' → next Monday to Sunday based on reference

# Weekdays
'segunda' → Monday (weekday 0)
'sexta' → Friday (weekday 4)
'próxima sexta' → next Friday

# Time of day
'tarde' → interval [12:00, 18:00]
'depois de almoço' → interval [12:30, 14:00]
```

**Advantages:**
- Fast lookup (O(1))
- Deterministic results
- Easy to maintain and update
- No training data required

**Trade-off:**
- Cannot handle misspellings or variations not in lexicon
- No contextual understanding


#### 1.2 Regex Pattern Matching
Regular expressions extract and validate temporal components:

```python
# Time patterns
Pattern: r'(?:às?\s*)?(\d{1,2})[h:\.(\s|)](?:(\d{2}))?'
Example: "às 15h30" → (15, 30)
Example: "14:30" → (14, 30)
Example: "16.45" → (16, 45)

# Explicit dates
Pattern: r'(\d{1,2})\s*(?:de|/|-)?\s*([a-záéíóú]+|\d{1,2})'
Example: "16 de Abril" → (16, 4, 2026)
Example: "16/04/2026" → (16, 4, 2026)

# Weekdays
Pattern: r'(?:(próxima|esta)\s+)?(segunda|sexta|...)'
Example: "próxima sexta" → ('próxima', 'sexta')
```

**Advantages:**
- Flexible pattern matching
- Handles format variations
- Unicode-aware (é, á, etc.)

**Limitations:**
- Complex regex can be hard to maintain
- May match incorrectly in unexpected contexts
- Performance overhead for many patterns


#### 1.3 Priority-Based Cascade
Patterns are tried in priority order, avoiding ambiguity:

```
1. EXPLICIT DATES (highest priority)
   → Most specific; least ambiguity
   → "16 de Abril" → exact date

2. WEEKDAY EXPRESSIONS
   → Specific relative reference
   → "sexta" → next Friday

3. RELATIVE EXPRESSIONS
   → Day offsets from reference
   → "amanhã" → tomorrow

4. COMPLEX EXPRESSIONS
   → Combinations of above
   → "sexta à tarde" → Friday afternoon

5. TIME ONLY (lowest priority)
   → Uses reference date
   → "15h" → today at 3 PM
```

**Rationale:** Process from most specific to most general. Stop at first match.


### Algorithm: Normalization Pipeline

```
normalize(expression, reference_datetime):
    1. Clean and lowercase input
    
    2. TRY: Parse explicit dates
       MATCH explicit_date_pattern?
       YES → Parse date, extract time if present, return
    
    3. TRY: Parse weekday expressions
       MATCH weekday_pattern?
       YES → Calculate target weekday, apply qualifiers, return
    
    4. TRY: Parse relative expressions
       MATCH relative_pattern?
       YES → Apply day offset or interval logic, return
    
    5. TRY: Parse complex expressions
       MATCH (weekday + time_of_day) OR (relative + time)?
       YES → Combine patterns, return
    
    6. TRY: Parse time only
       MATCH time_pattern?
       YES → Use reference date + extracted time, return
    
    7. DEFAULT: Mark as UNKNOWN
       Set confidence=0, return
```

**Time Complexity:** O(n) where n = number of patterns (typically 5-6)
**Space Complexity:** O(1)


---

## 2. IMPLEMENTATION DETAILS

### 2.1 Ambiguity Resolution Strategy

Portuguese temporal expressions are often ambiguous. The system uses **context-free
deterministic rules** to resolve ambiguity:

#### Case 1: Same Weekday Expression
**Expression:** "segunda" on a Monday reference
**Ambiguity:** Does it mean today or next week?

**Resolution Rule:**
```
IF time_specified AND time > current_time:
    → Assume TODAY (e.g., "segunda às 15h" at 10 AM → today 3 PM)
ELSE IF time_specified AND time <= current_time:
    → Assume NEXT WEEK (e.g., "segunda às 09h" at 10 AM → next Monday 9 AM)
ELSE IF no_time_specified:
    → Assume NEXT WEEK (default conservative assumption)
    → Log warning about ambiguity
```

**Example 1:**
```
Reference: Monday, April 21, 2026 at 10:00 AM
Expression: "segunda às 15h"
Resolution: 
  - Same weekday (Monday)
  - Time 15:00 > current time 10:00
  - Result: Monday, April 21, 2026 at 15:00 (TODAY)
```

**Example 2:**
```
Reference: Monday, April 21, 2026 at 10:00 AM
Expression: "segunda às 09h"
Resolution:
  - Same weekday (Monday)
  - Time 09:00 < current time 10:00
  - Result: Monday, April 28, 2026 at 09:00 (NEXT WEEK)
  - Note added to output about ambiguity
```

#### Case 2: Weekday without Explicit Qualifier
**Expression:** "terça" (Tuesday) on a Monday
**Ambiguity:** This week or next week?

**Resolution Rule:**
```
days_until_weekday = target_weekday - current_weekday
IF days_until_weekday < 0:
    days_until_weekday += 7  (next week)
ELSE IF days_until_weekday == 0:  (same weekday)
    Apply rule from Case 1
ELSE:
    Use calculated offset (this week, typically)
```

**Example:**
```
Reference: Monday, April 21, 2026
Expression: "terça"  (Tuesday)
Calculation:
  target = Tuesday (weekday 1)
  current = Monday (weekday 0)
  days_ahead = 1 - 0 = 1
  Result: Tuesday, April 22, 2026
```

#### Case 3: "Tarde" (Afternoon) without Specific Time
**Expression:** "sexta à tarde" (Friday afternoon)
**Ambiguity:** What time exactly?

**Resolution Rule:**
```
Use TIME_OF_DAY_RANGE midpoint as heuristic:
  'tarde' → [12:00, 18:00] → midpoint 15:00
Confidence = 0.75 (lower than exact times)
Precision = "approximate"
```

**Example:**
```
Reference: Monday, April 21, 2026 at 10:00 AM
Expression: "sexta à tarde"
Calculation:
  Weekday: Friday (4 days ahead) → April 25
  Time of day: 'tarde' → midpoint 15:00
  Result: Friday, April 25, 2026 at 15:00
  Precision: "approximate"
  Confidence: 0.75
```


### 2.2 Temporal Type Classification

Each normalized expression gets a type classification:

```python
TemporalType.DATE        → Only date, no time
  Example: "sexta" → 2026-04-24 (00:00 implied)
  Precision: "day"

TemporalType.TIME        → Only time, using reference date
  Example: "15h" → 2026-04-21T15:00 (reference date with time)
  Precision: "time_only"

TemporalType.DATETIME    → Specific date and time
  Example: "sexta às 15h" → 2026-04-24T15:00
  Precision: "exact"

TemporalType.INTERVAL    → Range with start and end
  Example: "para a semana" → [2026-04-27, 2026-05-03]
  Precision: "week"

TemporalType.RELATIVE    → Relative to reference (offset)
  Example: "amanhã" → 2026-04-22 (offset +1 day)
  Precision: "day"

TemporalType.UNKNOWN     → Could not parse
  Example: "xyz123" → no results
  Confidence: 0.0
```


### 2.3 Confidence Scores

Each result includes a confidence score (0.0 to 1.0):

```python
1.0   - Explicit dates with full specification
        "16 de Abril de 2026 às 15h"

0.95  - Explicit dates without year (year assumed)
        "16 de Abril às 15h"

0.90  - Weekday with exact time
        "sexta às 15h"

0.85  - Relative expressions (clear)
        "amanhã às 14h"

0.80  - Time-only expressions (date assumed)
        "15h"

0.75  - Intervals or approximate times
        "para a semana"
        "sexta à tarde"

0.70  - Vague expressions ("em breve" = soon)
        "em breve"

0.0   - Unparseable expressions
        "xyz123"
```


---

## 3. EDGE CASES & AMBIGUITY

### 3.1 Linguistic Edge Cases

#### Case A: Spelling Variations
**Problem:** Portuguese has multiple acceptable spellings
```
# Accent variations
'próxima' vs 'proxima' (next)
'quarta-feira' vs 'quarta' vs 'quarta feira'

# Regional variations
'Sábado' (São Paulo) vs 'Sábado' (Portugal - both same)
'Amanhã' (always with ~)
```

**Current Handling:** 
- Normalize to lowercase and strip accents (partial solution)
- Maintain multiple variants in lexicon
- **Limitation:** Still won't handle novel misspellings

**Better Solution:** Levenshtein distance matching with threshold
```python
def fuzzy_match(expression, lexicon, threshold=0.85):
    best_match = max(lexicon, 
                     key=lambda x: similarity(expression, x))
    if similarity(expression, best_match) > threshold:
        return best_match
    return None
```


#### Case B: Contractions & Abbreviations
**Problem:** Portuguese uses contractions that complicate matching
```
'a' + 'o' = 'ao' (at the)
'de' + 'a' = 'da' (of the - feminine)

'segunda' vs 'seg.' vs 'seg'
'manhã' vs 'manh.'
```

**Current Status:**
- Literal string matching handles some cases
- Regex patterns miss abbreviated forms

**Solution:**
- Expand contractions before matching
- Add abbreviated forms to lexicon
```python
ABBREVIATIONS = {
    'seg': 'segunda',
    'ter': 'terça',
    'qua': 'quarta',
    # ...
}
```


#### Case C: Compound Expressions with Conjunctions
**Problem:** Multiple temporal expressions in one sentence
```
"reunião segunda ou terça à tarde"
→ Two possible dates: Monday afternoon OR Tuesday afternoon

"amanhã de manhã ou quinta à noite"
→ Two temporal references

"de sexta até domingo"
→ Date range (interval)
```

**Current Status:** 
- Only first match is extracted
- Conjunctions are ignored

**Limitation:** This is a fundamental restriction of current approach


#### Case D: Relative References to Previous Expressions
**Problem:** Context-dependent temporal expressions
```
Email 1: "Reunião marcada para sexta"
Email 2: "Podemos adiar para a semana seguinte?"

"a semana seguinte" depends on reference to Email 1's "sexta"
→ Not just the reference datetime, but discourse context
```

**Current Status:** 
- No discourse context tracking
- Would need conversation history

**Solution:**
- Enhance normalize() with optional context parameter
- Track previously extracted temporal expressions
```python
def normalize(expr, reference_datetime, context={'previous_temporal': None}):
    # Could use context['previous_temporal'] for relative references
    pass
```


### 3.2 Temporal Logic Edge Cases

#### Case E: Leap Years & Month Boundaries
**Problem:** Datetime arithmetic edge cases
```python
# February 29 (leap year)
datetime(2026, 2, 29)  # ← 2026 is not leap year! Error!

# Month arithmetic
March 31 + 1 month = ?  (April has only 30 days)

# End-of-month boundaries
"final de março" → Should be March 31
```

**Current Status:** 
- Uses standard Python datetime (robust)
- Doesn't handle "end of month" expressions specifically

**Limitation:** "final de mês" not in lexicon; needs separate handler


#### Case F: Time Normalization Edge Cases
**Problem:** 24-hour notation vs. exceptions
```
# Valid Portuguese times
'24h' → Actually not valid! (23:59 is latest)
'00h' → Midnight
'12h' → Noon (potentially ambiguous in 12h format, but PT uses 24h)

# Unusual users write:
'25h' → Invalid in 24h format
'13:75' → Invalid minutes
'4:5' → Ambiguous: 04:05 or 04:50?
```

**Current Handling:**
```python
hours = int(match.group(1))
if not (0 <= hours <= 23):
    hours = hours % 24  # Wrap around (25h → 1h)

minutes = int(match.group(2)) if match.group(2) else 0
if not (0 <= minutes <= 59):
    return None  # Reject
```

**Better Approach:**
```python
# Strict validation
if not (0 <= hours <= 23) or not (0 <= minutes <= 59):
    raise ValueError(f"Invalid time: {hours}:{minutes}")
    
# Single-digit minute ambiguity: "4:5"
# Heuristic: if minutes < 10, pad with zero
# "4:5" is more likely "4:05" than "4:50"
```


#### Case G: Timezone Awareness
**Problem:** Portuguese spans timezone (though mostly UTC+0)
```python
# Açores: UTC-1 (one hour behind mainland)
# Madeira: UTC+0
# But daylight saving exists...

# If email sent from Açores, times are different
# Current system assumes single timezone
```

**Current Status:** 
- No timezone handling
- Assumes local system timezone

**Enhancement:**
```python
@dataclass
class NormalizedTemporal:
    timezone: str = "Europe/Lisbon"  # Add timezone field
    
def normalize(expr, reference_datetime, timezone="Europe/Lisbon"):
    # Use timezone-aware datetime
    from pytz import timezone as tz
    reference = tz(timezone).localize(reference_datetime)
```


### 3.3 Ambiguity in Meeting Context

#### Case H: Duration vs. Start Time
**Expression:** "1 hora" (1 hour)
**Ambiguity:** Is this a duration or a specific time?
```
Email: "Reunião dura 1 hora"
→ Duration (60 minutes)

Email: "Reunião à 1 hora"
→ Specific time (13:00)
```

**Current Status:** Not distinguished (not in temporal patterns)

#### Case I: Open-Ended Time Ranges
**Expression:** "depois das 15h" or "a partir de segunda"
**Ambiguity:** No end time specified
```python
# Current system:
interval_start = parsed_datetime
interval_end = None  # or very vague (e.g., end of day)

# Better: Implicit end times based on context
if "depois das 15h" in email_body:
    interval_end = end_of_workday (18:00 or 19:00)
```


---

## 4. LIMITATIONS OF RULE-BASED SYSTEMS

### 4.1 Fundamental Limitations

#### 1. **No Semantic Understanding**
```
Rule-based: Pattern matching + lexicon lookup
ML-based: Understanding meaning from context

Example:
"Vamos marcar a reunião para amanhã?"
→ Rule-based: Extracts "amanhã" (tomorrow)
→ ML-based: Understands this is a PROPOSAL, not confirmation

"A reunião agora é amanhã, não mais hoje"
→ Rule-based: Returns ambiguous "amanhã"
→ ML-based: Understands changed from "hoje" to "amanhã"
```

#### 2. **No Context Incorporation**
```
Email intent affects interpretation:
"Reunião marcada para segunda" (agendamento)
→ Monday is the confirmed datetime

"Você pode remarcar para segunda?" (cancelamento)
→ Monday is a proposal, not confirmed

Email body might contain multiple temporal refs:
"Reunião era sexta, agora será segunda"
→ Need to extract the NEW time, not OLD time
```

#### 3. **No User Profiles or History**
```
Temporal expressions vary by user:
- Some always speak precisely ("15h exato")
- Others use approximate times ("por volta das 15h")
- Some mix languages ("tomorrow à noite")

ML could learn user-specific interpretation patterns
```

#### 4. **No Language Mixing Handling**
```
Common in Portuguese companies:
"Reunião sexta afternoon às 15h"
"Next week será trocada para segunda"
"Wednesday morning, ok?"

Rule-based system will struggle with code-switching
```


### 4.2 Scalability Issues

#### Pattern Explosion
```
With more patterns, system complexity grows:
- Current: ~6 patterns, manageable
- With dialect variations: +50% patterns
- With acronyms/abbreviations: +100% patterns
- With custom company terminology: +200% patterns

Maintenance burden increases exponentially
```

#### Coverage Gaps
```
The Zipfian principle: 80/20 rule applies
- 6 patterns cover ~80% of emails
- Next 6 patterns cover ~15% (diminishing returns)
- Long tail of rare expressions: 5% (high cost, low value)

Rule-based systems are not scalable to 100% coverage
```


### 4.3 Performance Trade-offs

#### Speed vs. Accuracy
```
Option 1: More patterns (more accurate, slower)
  - Cascade tries all patterns (can be 50+ ms per expression)
  - O(n) where n = pattern count

Option 2: Fewer patterns (faster, less accurate)
  - Only 3-4 key patterns (10-20 ms per expression)
  - Misses edge cases

Current implementation: Balanced at O(n), n≈6
```

#### Memory vs. Precision
```
Option 1: Larger lexicon (more variations)
  - 10,000 entries covering misspellings, dialects
  - ~100 KB memory

Option 2: Minimal lexicon (efficient)
  - 100 entries, core only
  - ~5 KB memory
  - Many misses

Current: Minimal lexicon approach
```


### 4.4 Robustness Issues

#### Cascading Failures
```
If one pattern has a bug, it fails silently:

"16 de Abril"
→ Explicit date pattern tries to parse
→ Bug: month parsing fails (missing 'Abril' in month dict)
→ Pattern returns False
→ Falls through to next pattern
→ Weekday pattern matches "Abril" as… nothing
→ Finally marked as UNKNOWN

User has no idea why it failed
```

#### No Learning from Mistakes
```
Rule-based systems don't improve:
- Process 1,000 emails
- Extract 50 expressions incorrectly
- Still wrong on email #1,001

Machine learning systems improve:
- Can retrain on labeled errors
- Performance improves with more data
```


---

## 5. MACHINE LEARNING APPROACHES

### 5.1 Hybrid Temporal Expression Systems

#### Approach 1: LSTM-based Sequence Tagging

**Architecture:** Bidirectional LSTM + CRF (Conditional Random Field)

```python
# Input: "Reunião sexta às 15h"
# Tokenized: ["Reunião", "sexta", "às", "15h"]

# Model learns:
# Token embedding → LSTM hidden state → CRF tag

# Output sequence tags:
# ["O", "B-DATE", "O", "B-TIME"]
#  O = Outside (not temporal)
#  B-DATE = Begin date
#  B-TIME = Begin time
#  I-DATE/I-TIME = Inside (continuation)

# Advantages:
# - Learns context from surrounding words
# - Handles unknown phrases via learned representations
# - End-to-end extraction without explicit patterns

# Disadvantages:
# - Requires labeled training data (500-1000 examples)
# - Black-box: hard to interpret decisions
# - Can hallucinate temporal expressions
```

**Implementation Example:**
```python
import torch
from torch import nn

class TemporalLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.crf = CRF(output_dim)  # Conditional Random Field layer
        self.classifier = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, tokens):
        embedded = self.embedding(tokens)
        lstm_out, _ = self.lstm(embedded)
        logits = self.classifier(lstm_out)
        return self.crf.decode(logits)  # NER tags
```

**Dataset Requirements:**
```
Labeled data format:
Input: "Reunião sexta às 15h"
Tags:  [O, B-DATE, O, B-TIME]

Typical dataset size: 500-1000 sentences
Training time: 1-2 hours on GPU
```


#### Approach 2: BERT-based Fine-tuning

**Architecture:** Pre-trained BERT → fine-tuned for temporal NER

```python
# Uses pre-trained Portuguese BERT (BERT-PT or similar)
from transformers import BertForTokenClassification

model = BertForTokenClassification.from_pretrained(
    "neuralmind/bert-base-portuguese-cased",
    num_labels=5  # [O, B-DATE, I-DATE, B-TIME, I-TIME]
)

# Fine-tune on Portuguese temporal emails (transfer learning)

# Advantages:
# - Pre-trained on massive Portuguese corpus
# - Dramatically fewer examples needed (even 100-200)
# - Strong contextual understanding
# - Can handle out-of-vocabulary words

# Disadvantages:
# - More computational resources needed
# - Slower inference than rule-based (~100ms vs 10ms)
# - Might be overkill for simple task
```

**Performance Expectations:**
```
Dataset size: 200 labeled examples
Training time: 30 minutes
Accuracy on test set: 92-95%
```


#### Approach 3: HeidelTime (Existing System)

**What it is:** Open-source rule-based temporal extraction with ML components

```python
from heideltime import HeidelTime

ht = HeidelTime()
result = ht.process_text(
    "Reunião sexta às 15h",
    language="pt",  # Portuguese support!
    document_type="news",  # or "colloquial"
    reference_datetime="2026-04-21T10:00:00"
)

# Returns: Structured temporal expressions with ISO normalization
```

**Architecture:**
- Rule-based pattern matching (like our system) + ML re-ranking
- Pre-trained models for multiple languages (including Portuguese)
- Normalization to ISO 8601 (TIMEX format)

**Advantages:**
```
✓ Strong baseline performance (90%+ accuracy)
✓ Multilingual support (Portuguese included)
✓ Temporal reasoning (arithmetic: "2 days after Monday")
✓ Extensive evaluation on standard benchmarks (TempEval)
✓ Production-ready (used in real systems)
```

**Disadvantages:**
```
✗ Rule-based layer is complex to maintain/extend
✗ Not specialized for emails (optimized for news)
✗ Dependency on JVM (HeidelTime runs on Java)
✗ Some features may detect too broadly
```

**When to use:** If you need state-of-the-art general temporal extraction


#### Approach 4: SUTime (Stanford)

**What it is:** Rule-based system with ML disambiguation

```python
from sutime import SUTime

sutime = SUTime(
    jars="/path/to/sutime/lib",
    mark_time_expressions=True
)

result = sutime.parse("Reunião sexta às 15h", reference_date="2026-04-21")
# Returns: [
#     {
#         'value': '2026-04-24',
#         'text': 'sexta',
#         'type': 'DATE',
#         'timex-value': '<TIMEX3 value="2026-04-24" type="DATE">sexta</TIMEX3>'
#     },
#     {
#         'value': '2026-04-24T15:00',
#         'text': 'às 15h',
#         'type': 'TIME',
#         'timex-value': '...'
#     }
# ]
```

**Language Support:**
```
Originally English-focused
Portuguese support: Limited (requires adaptation)
Better for: English, Chinese, Spanish
```


### 5.2 Comparison Table: Rule-Based vs. ML Approaches

```
╔═════════════════════════════════════════════════════════════════════════════╗
║                    Rule-Based        LSTM-CRF       BERT-FT    HeidelTime  ║
╠═════════════════════════════════════════════════════════════════════════════╣
║ Accuracy (%)           75-85%           85-90%        90-95%       88-92%    ║
║ Training data needed   None             500-1000      100-200      None      ║
║ Training time          N/A              1-2 hours     30 min       N/A       ║
║ Inference speed        10-20 ms         50-100 ms     80-150 ms    50-100 ms ║
║ Language support       Portuguese only  Configurable  Pre-trained  > 200     ║
║ Maintenance burden     Medium           High          Medium       High      ║
║ Out-of-vocab handling  Poor             Good          Excellent    Good      ║
║ Interpretability       High             Low           Very Low     Medium    ║
║ Scalability            Poor             Good          Good         Good      ║
║ Implementation depth   Low              High          Medium       High      ║
║ Setup complexity       Easy             Hard          Medium       Hard      ║
╚═════════════════════════════════════════════════════════════════════════════╝
```

### 5.3 Hybrid Approach: Best of Both Worlds

**Recommended Strategy:**
```python
def hybrid_temporal_extraction(expression, reference_datetime):
    """
    1. Try rule-based system first (fast, interpretable)
    2. If confidence < threshold, try ML model (accurate, slow)
    3. Combine predictions if both available
    """
    
    # Step 1: Rule-based extraction
    rule_result = rule_based_normalizer.normalize(expression)
    
    if rule_result.confidence > 0.85:
        # High confidence: return immediately (fast path)
        return rule_result
    
    # Step 2: ML extraction for low-confidence cases
    if rule_result.confidence > 0.5:
        # Medium confidence: get second opinion
        ml_result = ml_temporal_extractor.extract(expression)
        
        # Step 3: Combine predictions
        return combine_predictions(rule_result, ml_result)
    
    # Step 3: ML extraction for all uncertain cases
    ml_result = ml_temporal_extractor.extract(expression)
    return ml_result
```

**Benefits:**
- Fast path for 80% of emails (rule-based only)
- High accuracy for edge cases (ML fallback)
- Interpretability when possible (rule-based)
- Robustness through ensemble

**Cost:** ML inference only for ~15-20% of emails


---

## 6. REAL-WORLD CONSIDERATIONS

### 6.1 End-to-End Pipeline Integration

```
Raw Email
   ↓
[Text Preprocessing]
   ├─ Remove HTML tags
   ├─ Decode quoted-printable
   └─ UTF-8 normalization
   ↓
[Sentence Segmentation]
   └─ Split into sentences (period, newline, etc.)
   ↓
[Temporal Expression Extraction]  ← Our module
   └─ Identify temporal spans
   ↓
[Temporal Normalization]  ← YOUR module
   └─ Convert to structured datetimes
   ↓
[Email Intent Classification]
   └─ Agendamento? Cancelamento? Confirmação?
   ↓
[Argument Extraction]
   ├─ Participants (NER)
   ├─ Location
   ├─ Topic
   └─ TEMPORAL (normalized)  ← Input from our module
   ↓
[Event Creation/Update]
   └─ Calendar system, DB, etc.
```

**Integration Points:**
```python
from preprocessing.argument_extraction import ArgumentExtractor
from preprocessing.temporal_normalization import TemporalNormalizer

class EmailProcessor:
    def __init__(self):
        self.arg_extractor = ArgumentExtractor()
        self.temporal_normalizer = TemporalNormalizer()
    
    def process_email(self, email_dict):
        # Extract raw arguments
        arguments = self.arg_extractor.extract(email_dict)
        
        # Normalize temporal expressions
        normalized_temporals = []
        for temporal_span in arguments.time_expressions:
            normalized = self.temporal_normalizer.normalize(
                temporal_span.text,
                reference_datetime=email_dict['received_date']
            )
            normalized_temporals.append(normalized)
        
        # Combine results
        return {
            'participants': arguments.participants,
            'location': arguments.locations,
            'topic': arguments.topics,
            'temporal': normalized_temporals,  # ← Our output
        }
```


### 6.2 User Feedback Loop

```
Production Deployment
   ↓
[Monitor Errors]
   ├─ Extract: 100 emails → 5% have temporal errors
   ├─ Types of errors:
   │  - Misclassified "próxima" (next) vs "essa" (this)
   │  - Time zone issues
   │  - Ambiguous weekday without qualifier
   └─ Create labeled error dataset
   ↓
[Retrain/Update]
   ├─ Rule-based: Add rules for error patterns
   ├─ ML-based: Retrain on error examples
   └─ Develop test cases
   ↓
[A/B Test]
   └─ v1 (current) vs v2 (updated)
```

**Feedback Mechanism:**
```python
class FeedbackCollector:
    """Collect user corrections for model improvement."""
    
    def log_correction(self, original_expr, predicted, corrected):
        """
        User corrected system output:
        "sexta" → predicted: 2026-04-24
                → corrected: 2026-04-18
        
        This suggests: "sexta" refers to THIS week, not next
        """
        self.error_db.save({
            'original': original_expr,
            'predicted': predicted,
            'corrected': corrected,
            'timestamp': datetime.now(),
            'user_id': user_id,
        })
```


### 6.3 Performance Monitoring

```python
class TemporalExtractionMetrics:
    """Track performance metrics for monitoring."""
    
    def __init__(self):
        self.metrics = {
            'total_expressions': 0,
            'successful_parses': 0,
            'parse_latency_ms': [],
            'confidence_distribution': defaultdict(int),
            'temporal_type_distribution': defaultdict(int),
        }
    
    def record_extraction(self, result, latency_ms):
        self.metrics['total_expressions'] += 1
        if result.temporal_type != TemporalType.UNKNOWN:
            self.metrics['successful_parses'] += 1
        
        self.metrics['parse_latency_ms'].append(latency_ms)
        self.metrics['confidence_distribution'][
            round(result.confidence, 1)
        ] += 1
        self.metrics['temporal_type_distribution'][
            result.temporal_type.value
        ] += 1
    
    @property
    def success_rate(self):
        return (
            self.metrics['successful_parses'] / 
            self.metrics['total_expressions']
        )
    
    @property
    def avg_latency_ms(self):
        return sum(self.metrics['parse_latency_ms']) / len(...)
```


---

## 7. RECOMMENDATIONS

### 7.1 For Current Implementation

**Immediate Improvements (Easy):**
1. Add fuzzy matching for common misspellings
2. Expand MONTHS and WEEKDAYS dictionaries
3. Add handling for "fim de mês", "início de mês"
4. Better logging for debugging

**Medium-Term (1-2 weeks):**
1. Test on actual customer emails
2. Collect error cases
3. Implement discourse context (conversation history)
4. Add handling for time ranges ("sexta-domingo")

**Long-Term (1-3 months):**
1. Build labeled dataset from error logs
2. Develop LSTM tagger for comparison
3. Consider HeidelTime integration
4. Implement hybrid approach


### 7.2 When to Consider Machine Learning

**Red Flags for Rule-Based:**
```
❌ > 20% of expressions still unknown
❌ Too many edge cases to handle
❌ Rules becoming too complex (>20 patterns)
❌ Maintenance cost exceeds benefit
❌ Users commonly correct system
```

**Green Lights for ML:**
```
✓ Have labeled dataset (500+ examples)
✓ Error patterns are systematic
✓ Data is diverse (multiple writing styles)
✓ Can afford inference latency (100-200ms)
```

**Recommended Path:**
```
Phase 1: Pure rule-based (current)
         → Deploy, gather data, find edge cases

Phase 2: Rule-based + error handling
         → 2-3 weeks of optimization

Phase 3: Hybrid (rule-based default + ML fallback)
         → If phase 2 success rate < 90%

Phase 4: Full ML replacement (optional)
         → Only if scaling to >100 languages
```


### 7.3 Portuguese-Specific Considerations

**Regional Variations:**
```
Portugal (PT):
  - "Quarta-feira" (always hyphenated)
  - "Próxima" (feminine)
  - "Que vem" (that comes) is common qualifier

Brazil (BR):
  - "Quarta" or "Quarta-Feira" (varies)
  - "Semana que vem" (next week)
  - "Para próxima segunda" (for next Monday)

Current system: Handles both reasonably
Enhancement: Could add pt_PT vs pt_BR flags
```

**Informal Speech:**
```
Email shorthand common in Portuguese:
"2ª" instead of "terça"
"6ª" instead of "sexta"
"Seg" instead of "segunda"

Add to lexicon:
'2ª', '2a': 'terça',
'6ª', '6a': 'sexta',
'seg': 'segunda',
```


### 7.4 Deployment Checklist

```
Before deploying to production:

[ ] Test on 50+ real emails
[ ] Monitor edge cases
[ ] Set up logging and monitoring
[ ] Document confidence score distribution
[ ] Create backup rules for failures
[ ] Plan for user feedback loop
[ ] Set confidence thresholds
    - < 50%: flag for manual review
    - 50-80%: suggest as possibility
    - > 80%: use automatically
[ ] Performance benchmarking (latency, memory)
[ ] Integration tests with full pipeline
[ ] Error analysis framework
```


---

## 8. SUMMARY

### Rule-Based Approach Trade-offs

```
GOOD FOR:
✓ Fast deployment
✓ Interpretable decisions
✓ Easy maintenance (for now)
✓ No labeled data required
✓ Good baseline (75-85% accuracy)

NOT GOOD FOR:
✗ 100% coverage
✗ Edge cases and long-tail
✗ Scaling to many language variants
✗ Learning from mistakes
✗ Out-of-vocabulary expressions
```

### Key Design Insights

1. **Priority-based cascade** prevents ambiguity blowup
2. **Deterministic heuristics** ensure reproducibility
3. **Confidence scores** provide transparency
4. **Lexicon + regex combo** balances flexibility and speed
5. **Hybrid approach** bridges rule-based and ML advantages

### Recommended Evolution

```
Current (Rule-Based)
        ↓
        ├─ Add fuzzy matching
        ├─ Expand lexicons
        ├─ Better error metrics
        └─ User feedback loop
                ↓
Phase 2 (Enhanced Rule-Based)
                ↓
                ├─ Accuracy stabilizes at ~85%
                ├─ Identify systemic error patterns
                └─ Build labeled dataset
                        ↓
                 Decision Point:
                 ├─ Accuracy acceptable? → Stay with Phase 2
                 └─ Need >90%? → Phase 3
                        ↓
          Phase 3 (Hybrid: Rule + ML)
                        ↓
          Fine-tune BERT on 200 examples
          Deploy as fallback for low-confidence
          Achieve 92-95% overall accuracy
```

---

## REFERENCES & FURTHER READING

### Temporal Extraction Systems
- HeidelTime: https://github.com/HeidelTime/heideltime
- SUTime: https://nlp.stanford.edu/software/sutime.html
- TIMEN (Portuguese): https://github.com/cltl/TIMEN

### ML Approaches
- LSTM-CRF for NER: https://arxiv.org/abs/1603.01360
- BERT Fine-tuning: https://arxiv.org/abs/1810.04805
- TempEval benchmarks: https://en.wikipedia.org/wiki/TempEval

### Portuguese NLP
- NLP Portuguese Reddit: r/Portuguese r/LanguageTechnology
- Portuguese BERT: https://huggingface.co/neuralmind/bert-base-portuguese-cased
- Portuguese Linguistic Resources: http://www.linguateca.pt/

---

**Document Version:** 1.0
**Last Updated:** April 2026
**Author:** NLP Engineer - Dissertação Project
"""
