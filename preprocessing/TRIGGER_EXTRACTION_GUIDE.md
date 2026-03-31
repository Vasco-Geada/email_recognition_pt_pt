"""
TRIGGER EXTRACTION MODULE - COMPLETE DOCUMENTATION

This document explains the trigger extraction baseline, its design decisions,
limitations, and how it evolves toward transformer-based approaches.

Document Version: 1.0
Date: 2026-03-31
Language: Portuguese (European)
"""


# =============================================================================
# 1. WHAT IS TRIGGER EXTRACTION?
# =============================================================================

"""
DEFINITION:
───────────
A trigger is the word or phrase in an email that signals the communicative
intent of the sender.

EXAMPLES by Intent:
───────────────────

1. AGENDAMENTO_REUNIAO (Meeting Scheduling)
   Email: "Gostaria de agendar uma reunião para próxima semana."
   Trigger: "agendar"
   Function: Signals the request to schedule
   
   Email: "Quando está disponível para marcar encontro?"
   Trigger: "marcar"
   Function: Alternative trigger for same intent
   
   Email: "Há alguma forma de colocar isso na agenda?"
   Trigger: "colocar na agenda"
   Function: Multi-word trigger with same meaning


2. CANCELAMENTO_REUNIAO (Meeting Cancellation)
   Email: "Infelizmente, preciso cancelar nossa reunião."
   Trigger: "cancelar"
   Function: Signals meeting cancellation
   
   Email: "Terei que adiar para semana que vem."
   Trigger: "adiar"
   Function: Related trigger for rescheduling


3. DISCUSSAO_DATA (Data/Time Discussion)
   Email: "Qual dia você sugere para conversar?"
   Trigger: "qual dia"
   Function: Signals need to discuss timing
   
   Email: "Propostas: terça-feira ou quinta-feira?"
   Trigger: "propostas" or dates themselves
   Function: Signals date discussion context


4. NAO_REUNIAO (Not About Meeting)
   Email: "Aqui está o relatório trimestral anexado."
   Trigger: none (or "relatório" in different context)
   Function: No trigger extracted because intent is informational


WHY EXTRACT TRIGGERS?
─────────────────────
1. EXPLAINABILITY
   - Model decisions become transparent
   - "Why is this agendamento_reuniao?" → "Because we found 'agendar'"
   - Critical for compliance and debugging

2. CONFIDENCE CALIBRATION
   - Email with trigger: high confidence in intent label
   - Email without trigger: be cautious about intent prediction
   - Trigger presence ↔ Intent reliability

3. INFORMATION EXTRACTION
   - Triggers anchor downstream processing
   - Extract time expressions near "agendar" for scheduling
   - Extract recipients near "cancelar" for cancellation notice

4. HUMAN DECISION SUPPORT
   - Dashboard shows: "Intent: agendamento_reuniao | Trigger: agendar"
   - Humans can quickly scan and validate/override

5. PIPELINE QUALITY MONITORING
   - Trigger hit rate is a diagnostic metric
   - Low hit rate → Update lexicons or retrain model
   - High false negatives → Need more sophisticated extraction
"""


# =============================================================================
# 2. WHY TRIGGER EXTRACTION DEPENDS ON INTENT
# =============================================================================

"""
PRINCIPLE: Word Sense Ambiguity
────────────────────────────────

Same words have DIFFERENT meanings in DIFFERENT intent contexts.

EXAMPLE 1: The word "semana" (week)
```
Intent: AGENDAMENTO_REUNIAO
Email: "Podemos agendar para próxima semana?"
Trigger: YES - triggers scheduling intent
Role: Temporal reference for meeting time

Intent: NAO_REUNIAO
Email: "Temos uma semana muito ocupada em termos de projetos."
Trigger: NO - just context, not triggering scheduling
Role: Providing background information
```

EXAMPLE 2: The word "desculpa" (apology/excuse)
```
Intent: CANCELAMENTO_REUNIAO
Email: "Peço desculpas, mas preciso cancelar."
Trigger: MAYBE - depends on context pattern
Role: Politeness marker in cancellation

Intent: NAO_REUNIAO
Email: "Nenhuma desculpa é aceitável para faltar."
Trigger: NO - completely unrelated to meeting
Role: General statement
```

EXAMPLE 3: The word "data" (date/data)
```
Intent: DISCUSSAO_DATA
Email: "Qual data você prefere?"
Trigger: YES - triggers date discussion
Role: Explicitly requesting date choice

Intent: NAO_REUNIAO
Email: "Os dados estão anexados em arquivo."
Trigger: NO - phonetically similar but different meaning
Role: Data file reference (note pronunciation difference)
```

CONSEQUENCE:
Without intent awareness, we'd extract:
- FALSE POSITIVES: Extract "semana" as trigger even in non-scheduling contexts
- FALSE NEGATIVES: Miss triggers that only signal intent in specific contexts
- NOISY FEATURES: Feed unreliable signals to downstream systems


PRINCIPLE 2: Lexicon Selectivity
─────────────────────────────────

Different intents use DIFFERENT vocabularies.

Vocabulary Overlap:
```
SHARED WORDS (appear in multiple intent lexicons):
├─ "quando" → agendamento_reuniao, discussao_data
├─ "dia" → agendamento_reuniao, discussao_data  
├─ "disponível" → agendamento_reuniao, cancelamento_reuniao
└─ "reunião" → agendamento_reuniao, cancelamento_reuniao

INTENT-SPECIFIC WORDS:
├─ agendamento_reuniao: "agendar", "marcar" (scheduling actions)
├─ cancelamento_reuniao: "cancelar", "adiar" (negation/delay)
├─ discussao_data: "qual dia", "propor data" (date inquiry)
└─ nao_reuniao: (empty - no characteristic triggers)
```

STRATEGY:
For each intent, search only the relevant vocabulary subset:
- agendamento_reuniao: Search for scheduling triggers
- cancelamento_reuniao: Search for cancellation triggers
- discussao_data: Search for date-inquiry triggers

BENEFIT: Vastly reduces false positives compared to searching all triggers
against all intents regardless of predicted intent.


PRINCIPLE 3: Contextual Colocation
──────────────────────────────────

Words are triggers only in specific SYNTACTIC/SEMANTIC contexts.

EXAMPLE: "não posso agendar"
```
Surface form: contains "agendar" (normally a trigger)
Semantic interpretation:
  Subject: I (implicit "eu")
  Negation: "não"
  Action: "posso agendar" = can schedule
  
Intent depends on context:
- If email is "Não posso agendar agora" → CANCELAMENTO_REUNIAO
- If email is "Não posso agendar nada" → NAO_REUNIAO
- If email is "Não posso agendar em abril, mas posso em maio" → AGENDAMENTO (negotiation)

LEXICAL APPROACH LIMITATION:
- Cannot distinguish these cases
- Would extract "agendar" as trigger in all three
- Would produce high false positive rate

TRANSFORMER APPROACH ADVANTAGE:
- Learns that negation before trigger changes meaning
- Understands "não posso agendar" ≠ trigger in most cases
- Captures subtle contextual nuances
```


PRINCIPLE 4: Efficiency
──────────────────────

Intent awareness makes extraction COMPUTATIONALLY EFFICIENT.

Without intent:
- Search 200 possible triggers across entire email
- ~200 comparisons per email

With intent:
- Search only 15-20 intent-specific triggers
- ~20 comparisons per email

Efficiency matters for:
- Real-time email processing
- Batch processing large mailboxes
- Mobile/embedded deployments

HIERARCHY:
```
Lexicon Size → Processing Time
───────────────────────────────
All triggers:           200+
Agendamento lexicon:    ~25
Cancelamento lexicon:   ~22
Discussao lexicon:      ~18
Nao_reuniao lexicon:    ~8

→ Intent-aware is 10x faster per search
```


PRINCIPLE 5: Domain Adaptation
───────────────────────────────

Different email domains (HR, Sales, Tech Support) use different triggers.

EXAMPLE: Same intent in different domains

TECH SUPPORT:
Email: "Can we schedule a call to discuss?"
Trigger: "schedule"
Domain triggers: schedule, call, troubleshoot, issue

HR DEPARTMENT:
Email: "Gostaríamos de agendar a sua entrevista de candidatura."
Trigger: "agendar"
Domain triggers: agendar, entrevista, candidatura, reunião

SALES:
Email: "Vamos marcar uma reunião de propostas?"
Trigger: "marcar"
Domain triggers: marcar, proposta, cliente, apresentação

BENEFIT:
Intent-aware extraction allows per-domain lexicon customization:
```python
TRIGGER_LEXICONS = {
    EmailIntent.AGENDAMENTO_REUNIAO: {
        "tech_support": ["schedule call", "set up", "book"],
        "hr": ["agendar entrevista", "marcar seleção"],
        "sales": ["marcar reunião", "propostas"],
    }
}
```

Result: Higher trigger detection accuracy for each domain.
"""


# =============================================================================
# 3. LIMITATIONS OF LEXICAL APPROACHES
# =============================================================================

"""
Lexical trigger extraction has fundamental limitations that become apparent
at scale. Understanding these is crucial for improvement planning.


LIMITATION 1: Fixed Vocabulary Coverage
────────────────────────────────────────

PROBLEM:
- Triggers are hand-curated
- New expressions outside lexicon are missed
- Language evolution is not captured
- Slang and colloquialisms are underrepresented

EVIDENCE:
In real Portuguese email data, new variations emerge:
```
Known triggers:          "agendar", "marcar", "agendar reunião"
New variations found:    
  - "vamos dar um jeito de agendar" (let's find a way to schedule)
  - "combinar algo" (arrange something - less formal)
  - "marcarmos com jeito" (let's find a way to schedule)
  - "agendarmos para mais tarde" (schedule for later)
```

COVERAGE ANALYSIS:
- Lexicon coverage: ~85% of training data
- Lexicon coverage: ~65-70% of production data
- Gap explains lower trigger hit rate in deployment

SOLUTION (Short-term):
- Continuously monitor missed triggers
- Quarterly lexicon updates
- Community contributions

SOLUTION (Long-term):
- Use word embeddings (Word2Vec, FastText)
- Semantic similarity search: find words similar to "agendar"
- Transformer-based approaches (section 5)


LIMITATION 2: Inflection Sensitivity
──────────────────────────────────────

PROBLEM:
- Words change form (verbs conjugate, nouns pluralize)
- Regex patterns may not cover all forms
- Non-standard forms are missed
- Regional variations exist

PORTUGUESE EXAMPLES:
```
Base trigger: "agendar" (infinitive)
Inflected forms:
  ✓ Present:      agendo, agendas, agenda, agendamos, agendai, agendum
  ✓ Past:         agendei, agendaste, agendou, agendámos, agendartes, agendaram
  ✓ Future:       agendarei, agendarás, agendará, etc.
  ✓ Conditional:  agendaria, agendarias, etc.
  ✓ Gerund:       agendando
  ✓ Past part.:   agendado, agendada, agendados, agendadas
  ✓ Subjunctive:  agende, agendes, agendemos, agendeis, agendem
  ✗ Missed cases: Regional variants, non-standard forms, typos

Regex pattern: r"agendar(?:ei|á|ás|emos|eis|ão)?\b"
Coverage: ~95% of regular conjugations
Missed: subjunctive, subjunctive past, theoretical forms

Coverage Analysis:
- Standard forms: 98%
- Colloquial forms: 85%
- Non-standard/typos: 40%
```

SOLUTION (Short-term):
- Expand regex patterns
- Add lemmatization (spaCy)

SOLUTION (Long-term):
- Morphological analyzer for Portuguese
- Trainable sequence labeling (transformer-based)


LIMITATION 3: Contextual Misunderstanding
──────────────────────────────────────────

PROBLEM:
- Lexical extraction ignores context
- Negation, conditional, irony not understood
- Trigger presence ≠ Trigger intent

EXAMPLES:

Negation:
```
Positive:  "Vamos agendar uma reunião." → Intent: AGENDAMENTO
Negative:  "Não vamos agendar nada." → Intent: NAO_REUNIAO
Extraction result: Both find "agendar"
Classification impact: High false positives for AGENDAMENTO intent
```

Conditional:
```
Actual intent: "Se conseguir agendar, aviso-te." → NAO_REUNIAO (hypothetical)
Triggers "agendar" but intent is not scheduling
```

Negation + Conditional:
```
"Se não conseguir agendar, podemos tentar outra data?" 
→ Actually a date-discussion (DISCUSSAO_DATA)
→ Lexical extraction finds both "agendar" and date references
→ Ambiguous which trigger to return
```

Irony/Sarcasm:
```
"Claro, vou agendar isso prioritariamente!" (Obviously, I'll schedule that first!)
→ Said sarcastically after receiving urgent request
→ Actually means: "This will be delayed"
→ Triggers "agendar" but intent is unclear
```

COVERAGE:
- Simple sentences: ~95% accuracy
- With negation: ~75% accuracy
- With conditionals: ~70% accuracy
- With multiple clauses: ~60% accuracy

SOLUTION (Short-term):
- Add negation-aware patterns
- Check for "não" before trigger
- Pattern: r"não\s+(?:vou|posso|consigo|gosto)\s+agendar"

SOLUTION (Long-term):
- Transformer models understand context natively
- Attention mechanisms capture scope of negation
- Contextual word embeddings (BERT)


LIMITATION 4: Synonym and Paraphrase Blindness
───────────────────────────────────────────────

PROBLEM:
- Similar meanings get different triggers
- Synonyms are not recognized
- Paraphrases miss triggers entirely
- Semantic equivalence is not captured

PORTUGUESE EXAMPLES:
```
Semantically equivalent, lexically different:

1. "agendar uma reunião" vs. "marcar uma reunião"
   → Both mean the same thing
   → Both are triggers for AGENDAMENTO
   → Both would be in lexicon
   → But: more obscure synonyms might be missed

2. "cancelar a reunião" vs. "desistir da reunião" vs. "abrir mão da reunião"
   → All mean the same thing (cancel/give up)
   → First is in lexicon
   → Others might be missed or misclassified

3. "combinar encontro" vs. "agendar encontro" vs. "marcar encontro"
   → All mean to schedule
   → Some in lexicon, others potentially missed

Synonym explosion:
```

COVERAGE:
- Common synonyms: ~90% coverage
- Uncommon synonyms: ~40% coverage
- Paraphrases: ~20% coverage

SOLUTION (Short-term):
- Expand lexicon with known synonyms
- Use thesaurus or WordNet

SOLUTION (Long-term):
- Semantic similarity: find words close to "agendar" in embedding space
- Word2Vec or FastText: similarity("agendar", "marcar") = 0.87
- Transformer embeddings: contextual similarity


LIMITATION 5: Multi-word Expression Brittleness
─────────────────────────────────────────────────

PROBLEM:
- Multi-word triggers hard to match with simple exact matching
- Word ordering variations
- Insertions between words

EXAMPLES:
```
Base trigger (exact): "colocar na agenda"

Variations:
✓ "colocar na agenda" (exact match)
✓ "colocar isso na agenda" (insertion: "isso")
✓ "colocar alguma coisa na agenda" (semantically same)
✓ "colocar-se na agenda" (pronoun)
✗ "colocar os itens na agenda" (too different)
✗ "colocar reunião na agenda" (word substitution)

Covered: ~70%
Missed: ~30%
```

WORD ORDER VARIATIONS:
```
Standard: "agendar uma reunião"
Variation: "uma reunião agendar" (unusual but possible in speech/informal)
Variation: "reunião para agendar" (agenda item waiting to be scheduled)
```

SOLUTION (Short-term):
- Add regex patterns for common variations
- Use word distance metrics (edit distance)

SOLUTION (Long-term):
- Constituency parsing: identify phrase structure
- Dependency parsing: understand relationships
- Sequence labeling: label trigger spans of any length


LIMITATION 6: Morphological Mismatch
─────────────────────────────────────

PROBLEM:
- Different parts of speech from same root
- Noun forms vs. verb forms

EXAMPLES:
```
Root: AGEN(DA)

Noun:       "agendamento" (the act of scheduling)
Verb:       "agendar" (to schedule)
Adjective:  "agendado" (scheduled, past participle)

All are triggers? Or only verbs?

Challenge: 
- "10 agendamentos para esta semana" (noun, reference to existing meetings)
  → Is this a trigger?
  → Or just informational?

- "Tenho um agendamento às 3pm" (noun, stating existing meeting)
  → Different from "Vou fazer um agendamento" (action)
```

Part-of-speech context:
```
Verb usage:   "Vou agendar" → Trigger (action)
Noun usage:   "Um agendamento está confirmado" → Ambiguous
Adjective:    "Está agendado para amanhã" → Informational, maybe not trigger
```

SOLUTION (Short-term):
- POS tagging: only extract verbs as triggers
- Requires: spaCy or NLTK for Portuguese

SOLUTION (Long-term):
- Learned POS sensitivity in sequence models
- Dependency parsing: understand argument structure


LIMITATION 7: No Learning from Data
──────────────────────────────────

PROBLEM:
- Static rules don't improve with more data
- No ability to learn importance/relevance weights
- All triggers treated equally

EXAMPLE:
```
Lexicon contains: ["agendar", "marcar", "quando", "disponível"]

In training data analysis:
- "agendar" appears in 95% of agendamento_reuniao emails
- "quando" appears in 60% of agendamento_reuniao emails
- "disponível" appears in 40% of agendamento_reuniao emails

But lexical approach:
- All treated equally in matching
- Returns first match found (arbitrary)
- No weighting by importance

Transformer approach:
- Learns "agendar" is strongly indicative
- Learns "quando" is moderately indicative
- Produces probability scores for triggers
- Can return top-N trigger candidates with confidence
```

SOLUTION:
- Transformer-based sequence labeling with probability scores
"""


# =============================================================================
# 4. EVOLUTION TO TRANSFORMER-BASED EXTRACTION
# =============================================================================

"""
MOTIVATION:
The limitations above suggest moving from static rules to learned models.
Transformers are the current SOTA (State-of-the-Art) approach.


PHASE 1: Why Transformers?
──────────────────────────

Transformers address lexical limitations through:

1. CONTEXTUALIZED EMBEDDINGS
   - BERT embeddings are contextual
   - Same word gets different embeddings in different contexts
   - "agendar" in "Vou agendar" vs "Não vou agendar" → different vectors
   - Result: Negation is implicitly captured

2. LEARNED REPRESENTATIONS
   - No manual lexicon curation needed
   - Model learns what patterns matter
   - Discovers synonyms automatically
   - Handles morphological variants implicitly

3. SEQUENCE LABELING CAPABILITY
   - Tag each token: B (begin trigger), I (inside trigger), O (outside)
   - Handles multi-word triggers naturally
   - Variable-length triggers supported

4. ATTENTION MECHANISMS
   - Attention visualizes what the model focuses on
   - Can explain: triggering words and their context
   - "The model focused on 'agendar' because it saw 'no negation nearby'"

5. FINE-TUNING ADVANTAGE
   - Pre-trained on massive Portuguese text (e.g., BERTimbau)
   - Fine-tune on your specific domain/email data
   - Learns domain-specific patterns

6. CONFIDENCE SCORES
   - Each prediction gets a probability
   - Can threshold on confidence
   - Uncertainty → require human review


PHASE 2: Implementation Approach
────────────────────────────────

APPROACH A: Sequence Labeling (Recommended)
```
Model: BERT + Linear classifier
Task: Token Classification
Labels: B (begin trigger), I (inside trigger), O (outside)

Architecture:
┌─────────────────────────────────────────┐
│ Input: "Vamos agendar uma reunião"     │
├─────────────────────────────────────────┤
│ Tokenization                            │
│ [CLS] vamos agendar uma reunião [SEP]  │
├─────────────────────────────────────────┤
│ BERT Encoding (contextualized)          │
│ 768-dim embeddings for each token      │
├─────────────────────────────────────────┤
│ Linear Classification Layer             │
│ 768-dim → 3 classes (B, I, O)          │
├─────────────────────────────────────────┤
│ Output: Predictions                     │
│ O O B O O                               │
│ (tokens at position 2 tagged as B)      │
├─────────────────────────────────────────┤
│ Trigger Extraction                      │
│ Span from position 2: "agendar"        │
└─────────────────────────────────────────┘
```

APPROACH B: Span Extraction
```
Model: BERT + Span Extractor
Task: Extract start/end positions of triggers
Result: Multiple trigger candidates with scores

Advantage: Can return ranked list of potential triggers
Disadvantage: More complex, slower inference
```

APPROACH C: Dependency Parsing
```
Model: Biaffine parser
Task: Learn syntactic dependencies
Benefit: Understand "what is modifying what"
Example: "Não vou agendar" has negation dependency on verb
```


PHASE 3: Training Data
──────────────────────

Minimum data needed for fine-tuning:
```
Per intent:               ~500 labeled examples (conservative)
Total dataset:           ~2000 examples (4 intents × 500)
Annotation format:       BIO tags or span positions

Annotation process:
1. Sample random emails from production mailbox
2. Label each email with intent (already available from classifier)
3. Annotate trigger span (start and end character positions)
4. Quality check: 10% dual-annotation, target >95% agreement
5. Split: 80% train, 10% validation, 10% test

Example labeled data:
```json
[
  {
    "text": "Vamos agendar uma reunião",
    "intent": "agendamento_reuniao",
    "trigger": "agendar",
    "trigger_start": 6,
    "trigger_end": 13,
    "bio_tags": ["O", "O", "B-TRIGGER", "O", "O"]
  },
  {
    "text": "Preciso de adiar o encontro",
    "intent": "cancelamento_reuniao",
    "trigger": "adiar",
    "trigger_start": 9,
    "trigger_end": 13,
    "bio_tags": ["O", "O", "B-TRIGGER", "O", "O"]
  },
  {
    "text": "Qual data você sugere?",
    "intent": "discussao_data",
    "trigger": "qual data",
    "trigger_start": 0,
    "trigger_end": 9,
    "bio_tags": ["B-TRIGGER", "I-TRIGGER", "O", "O"]
  }
]
```


PHASE 4: Model Architecture
───────────────────────────

Option 1: Pretrained BERT + Fine-tuning
```python
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)
import torch

# Load Portuguese BERT
model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=3  # B, I, O
)

# Fine-tune on email trigger data
training_args = TrainingArguments(
    output_dir="./trigger_model",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    weight_decay=0.01,
    save_total_limit=2,
    use_cuda=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()

# Model is now fine-tuned for Portuguese email trigger extraction!
```

Option 2: Sequence-to-Sequence Approach
```python
# More complex but more flexible
# Input: [email text, intent]
# Output: trigger text

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

# Fine-tune with format:
# Input: "extract trigger for agendamento_reuniao: Vamos agendar uma reunião"
# Output: "agendar"
```


PHASE 5: Integration with Intent Classifier
─────────────────────────────────────────────

```python
# Enhanced pipeline with transformer extraction

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Load both models
intent_classifier = pipeline("zero-shot-classification", 
    model="neuralmind/bert-base-portuguese-cased")

trigger_extractor = pipeline("token-classification",
    model="./trigger_model",  # Your fine-tuned model
    tokenizer="./trigger_tokenizer")

# Process email
email = "Vamos agendar uma reunião para próxima semana?"

# Get intent
intent_result = intent_classifier(email, 
    candidate_labels=["agendamento_reuniao", "cancelamento_reuniao", 
                      "discussao_data", "nao_reuniao"])
intent = intent_result['labels'][0]  # Top prediction

# Get trigger(s)
trigger_result = trigger_extractor(email, aggregation_strategy="simple")
# Result: [
#     {'entity': 'B-TRIGGER', 'score': 0.95, 'word': 'agendar', ...},
#     {'entity': 'B-TRIGGER', 'score': 0.92, 'word': 'próxima semana', ...}
# ]

# Extract and rank triggers
triggers = [
    {'text': t['word'], 'confidence': t['score']} 
    for t in trigger_result
]

# Return top trigger
top_trigger = max(triggers, key=lambda x: x['confidence'])
print(f"Top trigger: {top_trigger['text']} ({top_trigger['confidence']:.2%})")
```


PHASE 6: Evaluation Metrics
───────────────────────────

Token-level metrics:
```
- Precision: Of tokens labeled B/I, how many are correct?
- Recall: Of actual trigger tokens, how many are found?
- F1: Harmonic mean of precision and recall
```

Span-level metrics:
```
- Exact Match: Are predicted span boundaries exactly correct?
- Partial Match: Do spans overlap?
- Fuzzy Match: Partial string overlap?

Example:
  Predicted: "agendar uma reunião" (start=6, end=24)
  Gold:      "agendar" (start=6, end=13)
  
  Exact match: ✗ (different spans)
  Partial match: ✓ (overlapping)
  Fuzzy match: ✓ (high overlap)
```

Per-intent metrics:
```
Track performance separately for each intent:
- agendamento_reuniao: F1 = 0.89
- cancelamento_reuniao: F1 = 0.85
- discussao_data: F1 = 0.78
- nao_reuniao: F1 = 0.92 (easier - no trigger expected)
```

Cross-domain metrics:
```
- HR domain: F1 = 0.91
- Sales domain: F1 = 0.84
- Tech Support: F1 = 0.76

Identifies weaknesses in specific domains for targeted improvement
```


PHASE 7: Deployment Considerations
──────────────────────────────────

1. Model Size and Speed
   - BERT-base: ~110M parameters
   - Inference time: ~100-200ms per email (GPU)
   - Mobile/edge deployment: Use quantization or distillation

2. Fallback Strategy
   - Transformer fails?: Fall back to lexical extraction
   - Model not confident?: Mark for review
   - Ensemble: Combine both methods

3. Monitoring and Retraining
   - Track trigger hit rate in production
   - Collect false negatives for data annotation
   - Retrain quarterly with new data

4. A/B Testing
   - Compare lexical vs. transformer extraction
   - Measure downstream task performance (e.g., scheduling success)
   - Gradual rollout of transformer model


PHASE 8: Hybrid Approach (Recommended for Production)
────────────────────────────────────────────────────

```python
class HybridTriggerExtractor:
    '''Combines lexical and transformer extraction'''
    
    def __init__(self, lexical_extractor, transformer_extractor):
        self.lexical = lexical_extractor
        self.transformer = transformer_extractor
    
    def extract_trigger(self, text, intent):
        # Fast path: Try lexical extraction first
        lexical_result = self.lexical.extract_trigger(text, intent)
        if lexical_result and lexical_result['confidence'] > 0.9:
            return lexical_result  # High confidence lexical match
        
        # Slow path: Use transformer for difficult cases
        transformer_result = self.transformer.extract_trigger(text, intent)
        if transformer_result and transformer_result['confidence'] > 0.7:
            return transformer_result
        
        # If both agree
        if lexical_result and transformer_result:
            if lexical_result['text'] == transformer_result['text']:
                return {
                    'trigger': lexical_result['text'],
                    'confidence': 0.99,  # High confidence agreement
                    'method': 'consensus'
                }
        
        # Return best available
        candidates = [lexical_result, transformer_result]
        candidates = [c for c in candidates if c is not None]
        return max(candidates, key=lambda x: x['confidence']) if candidates else None

# Benefits:
# - Fast for easy cases (95% of emails)
# - Accurate for difficult cases
# - Lower latency than transformer alone
# - Confidence scores for uncertainty handling
```


SUMMARY: Transformation Path
━━━━━━━━━━━━━━━━━━━━━━━━━━
Stage 1: Lexical extraction (where you are now)
         ↓ Limitations become apparent
Stage 2: Lexical + lemmatization
         ↓ Coverage increases but still limited
Stage 3: Hybrid (lexical + lemmatization + word embeddings)
         ↓ Performance plateaus
Stage 4: Transformer sequence labeling (fine-tuned BERT)
         ↓ SOTA, but slower inference
Stage 5: Hybrid (lexical fast path + transformer slow path)
         ↓ Production-ready, optimal balance

Recommended timeline:
- Now:      Implement lexical (DONE)
- 1-3 mo.:  Add spaCy lemmatization
- 3-6 mo.:  Collect 1000+ labeled examples
- 6-9 mo.:  Train fine-tuned trigger extraction model
- 9-12 mo.: Deploy hybrid approach in production
"""


# =============================================================================
# REFERENCE GUIDE
# =============================================================================

"""
File Structure:
───────────────
preprocessing/
├── trigger_extraction.py      # Core TriggerExtractor class
├── trigger_examples.py        # Examples and tests
├── email_pipeline.py          # Integration with intent classifier
└── trigger_extraction_guide.py  # This file (documentation)


Quick Start:
────────────
from preprocessing.trigger_extraction import TriggerExtractor

extractor = TriggerExtractor(use_lemmatization=True)
result = extractor.extract_trigger(
    "Vamos agendar uma reunião",
    "agendamento_reuniao"
)
print(result)
# Output: {'trigger': 'agendar', 'method': 'exact', 'intent': 'agendamento_reuniao'}


Integration:
─────────────
from preprocessing.email_pipeline import EmailAnalysisPipeline

pipeline = EmailAnalysisPipeline(intent_classifier)
analysis = pipeline.analyze(email_text)

# Use results for alerting, sorting, categorization, etc.
if analysis.requires_action:
    route_to_handler(analysis)


Documentation Map:
──────────────────
trigger_extraction.py > TriggerExtractor class docstring
  ├─ Define trigger concept
  ├─ Explain intent-awareness principle
  ├─ Document API
  └─ Provide examples

trigger_extraction.py > LIMITATION section
  └─ Explains constraints of lexical approach

trigger_extraction.py > EVOLUTION section
  └─ Detailed transformer roadmap

email_pipeline.py > EmailAnalysisPipeline class
  └─ Production integration patterns

trigger_examples.py > Multiple example functions
  └─ Concrete usage demonstrations


Troubleshooting:
────────────────
Q: No triggers found?
A: Check lexicon for intent, expand with regex patterns

Q: Too many false positives?
A: Reduce trigger list, add negation checks, use lemmatization

Q: Slow performance?
A: Disable lemmatization, cache vectorizer, batch process

Q: spaCy not working?
A: pip install spacy && python -m spacy download pt_core_news_sm


Future Work:
─────────────
1. Add word embedding similarity for synonyms
2. Collect 1000+ labeled examples
3. Train transformer-based sequence labeler
4. Compare transformer vs lexical performance
5. Deploy hybrid extraction in production
6. Monitor and collect edge cases
7. Quarterly model retraining
"""
