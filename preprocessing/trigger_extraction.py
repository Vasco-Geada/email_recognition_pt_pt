"""
Trigger Extraction Module for Event-Related Email Classification

This module provides a baseline lexical trigger extractor for identifying
event-related trigger words/phrases in Portuguese email texts, with support for
intent-specific lexical resources and regex-based pattern matching.

Author: Email Recognition System
Date: 2026
"""

import re
import logging
from typing import Optional, Dict, List, Tuple
from enum import Enum


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailIntent(Enum):
    """Email intent class enumeration."""
    AGENDAMENTO_REUNIAO = "agendamento_reuniao"
    CANCELAMENTO_REUNIAO = "cancelamento_reuniao"
    REUNIAO_CONFIRMADA = "reuniao_confirmada"
    NAO_REUNIAO = "nao_reuniao"


class TriggerExtractor:
    """
    Extracts trigger words/phrases from email text based on predicted intent.
    
    A trigger is the word or phrase that activates/signals the event intent.
    For example, "agendar" is a trigger for agendamento_reuniao.
    
    Advantages of intent-aware trigger extraction:
    1. **Contextual Precision**: Same word has different meanings in different contexts
       - "marcar" (mark/schedule) → trigger in agendamento, not in nao_reuniao
       - "cancelar" → trigger in cancelamento, neutral in others
    
    2. **Reduced False Positives**: Word frequency bias is mitigated
       - Without intent awareness, common words would generate noise
       - With intent, only relevant triggers are extracted
    
    3. **Better Generalization**: Intent-specific resources can be tuned
       - Different intents may require different trigger vocabularies
       - Allows for domain-specific customization
    
    4. **Interpretability**: Decisions are explicable
       - "I extracted 'agendar' because intent is agendamento_reuniao"
       - Valuable for debugging and model auditing
    """
    
    # Intent-specific trigger lexicons
    # Each intent has a curated set of trigger patterns
    TRIGGER_LEXICONS = {
        EmailIntent.AGENDAMENTO_REUNIAO: {
            "exact": [
                "agendar",
                "marcar",
                "agendar reunião",
                "marcar reunião",
                "agendar encontro",
                "marcar encontro",
                "falar",
                "reunião",
                "agendamento",
                "marque",
                "agendai",
                "marcai",
                "marca",
                 "quando",
                "qual dia",
                "que dia",
                "qual data",
                "que data",
                "próxima semana",
                "próximo mês",
                "amanhã",
                "depois de amanhã",
                "hoje",
                "semana que vem",
                "data",
                "sugerir data",
                "propor data",
                "próxima segunda",
                "próxima terça",
                "próxima quarta",
                "próxima quinta",
                "próxima sexta",
                "próximo sábado",
                "próximo domingo",
            ],
            "regex": [
                r"\bagendar(?:ei|á|ás|emos|eis|ão)?\b",  # agendar + verbal forms
                r"\bmarcar(?:ei|á|ás|emos|eis|ão)?\b",   # marcar + verbal forms
                r"\bquando\s+(?:pode|consegue|está)",     # quando pode, consegue
                r"\bdisponí?vel\b",                        # disponível/disponevel
                r"\bhorário\b",                            # horário (time slot)
                r"\breunião\b",                            # horário (time slot)
                r"\bslot\b",                               # slot (time slot)
                r"\bcolocar\s+na\s+agenda\b",              # colocar na agenda
                r"\bagendar\s+para\b",                     # agendar para
                 r"\bquando\b",                               # quando
                r"\b(?:qual|que)\s+(?:dia|data)\b",        # qual/que dia/data
                r"\b(?:próximo|próxima)\s+(?:semana|mês)\b", # próximo(a) semana/mês
                r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",       # data format DD/MM/YYYY
                r"\b(?:segunda|terça|quarta|quinta|sexta|sábado|domingo)\b", # weekdays
                r"\bdata\b",                                 # data
                r"\bque\s+tal\b",                            # que tal
            ],
            "lemma": ["agendar", "marcar", "agendamento"],
        },
        EmailIntent.CANCELAMENTO_REUNIAO: {
            "exact": [
                "cancelar",
                "cancelamento",
                "não posso",
                "indisponível",
                "impossível",
                "adiar",
                "adiamento",
                "postpone",
                "postponed",
                "reschedule",
                "lamento",
                "desculpa",
                "não me dá jeito",
                "ocupado",
            ],
            "regex": [
                r"\bcancelar(?:ei|á|ás|emos|eis|ão)?\b",  # cancelar + verbal forms
                r"\badiar(?:ei|á|ás|emos|eis|ão)?\b",     # adiar + verbal forms
                r"\bcancelado\b",                          # cancelado (past participle)
                r"\b(?:não|nao)\s+(?:vou|posso|consigo)\b", # não vou/posso/consigo
                r"\bindisponí?vel\b",                      # indisponível
                r"\bconflito\s+de\s+agenda\b",             # conflito de agenda
                r"\bdesculpa\b",                           # desculpa
                r"\blamento\b",                            # lamento
                r"\bocupado\b",                            # ocupado
                r"\b(?:não|nao|n)\s+(?:me\s+dá|dá|da)\s+(?:jeito|saída)\b", # não me dá jeito
            ],
            "lemma": ["cancelar", "adiar", "cancelamento", "adiamento"],
        },
        EmailIntent.REUNIAO_CONFIRMADA: {
            "exact": [
               "confirmar",
                "confirmado",
                "fica para",
                "confirmamos",
                "lá estarei",
                "está confirmado",
                "confirmo presença",
                "confirmo que estarei",
                "vou",
                "posso",
                "consigo",
                "estarei presente",
                "estarei lá",
                "pode ser",
                "por mim tudo bem",
                "sem problema",
            ],
            "regex": [
                 r"\bestar(?:ei|á|ás|emos|eis|ão)?\b",  # estar + verbal forms
                 r"\bconfirm(?:ar|ado|amos)?\b",  # confirmar + verbal forms
                 r"\bvou\b",  # vou
                 r"\bpor\s+mim\b",  # vou
                 r"\bconsigo\b",  # consigo
                 r"\bpor\s+mim\s+(?:posso|consigo|tudo\s+bem|pode\s+ser)\b",  # pode ser
                 
            ],
            "lemma": ["confirmar", "consigo", "estar", "estarei", "confirmo"],
        },
        EmailIntent.NAO_REUNIAO: {
            "exact": [
                "não é sobre reunião",
                "não é reunião",
                "off topic",
                "fora do assunto",
                "assunto diferente",
            ],
            "regex": [
                r"\b(?:não|nao)\s+(?:é|eh)\s+(?:sobre\s+)?(?:reunião|reuniao)\b",
                r"\b(?:off\s+topic|off-topic)\b",
                r"\b(?:fora\s+do|out\s+of)\s+(?:assunto|scope)\b",
                r"\bassunto\s+(?:diferente|outro)\b",
            ],
            "lemma": ["não", "reunião"],
        },
    }
    
    def __init__(self, use_lemmatization: bool = False, language: str = "pt"):
        """
        Initialize the TriggerExtractor.
        
        Args:
            use_lemmatization: If True, attempts to use spaCy lemmatization
                              for Portuguese. Requires: pip install spacy
                              and: python -m spacy download pt_core_news_sm
            language: Language code (currently only 'pt' for Portuguese)
        
        Raises:
            ValueError: If language is not supported
        """
        if language != "pt":
            raise ValueError(f"Language '{language}' not supported. Only 'pt' is available.")
        
        self.use_lemmatization = use_lemmatization
        self.language = language
        self.nlp = None
        
        if use_lemmatization:
            try:
                import spacy
                self.nlp = spacy.load("pt_core_news_sm")
                logger.info("spaCy Portuguese model loaded successfully")
            except ImportError:
                logger.warning(
                    "spaCy not installed. Install with: pip install spacy\n"
                    "Then download Portuguese model with: python -m spacy download pt_core_news_sm"
                )
                self.use_lemmatization = False
            except OSError:
                logger.warning(
                    "Portuguese spaCy model not found. Download with:\n"
                    "python -m spacy download pt_core_news_sm"
                )
                self.use_lemmatization = False
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for matching.
        
        Operations:
        - Convert to lowercase
        - Remove extra whitespace
        - Normalize common accents/diacritics variations
        
        Args:
            text: Raw email text
            
        Returns:
            Normalized text
        """
        text = text.lower().strip()
        # Normalize variations like "não" vs "nao"
        text = text.replace("ã", "a").replace("õ", "o").replace("ç", "c")
        text = re.sub(r"\s+", " ", text)  # Collapse multiple spaces
        return text
    
    def _get_lemmas(self, text: str) -> List[str]:
        """
        Extract lemmas from text using spaCy.
        
        Args:
            text: Normalized text
            
        Returns:
            List of lemmas from the text
        """
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text[:1000000])  # Limit to prevent memory issues
            return [token.lemma_ for token in doc]
        except Exception as e:
            logger.warning(f"Lemmatization failed: {e}")
            return []
    
    def _match_exact_trigger(
        self,
        text: str,
        triggers: List[str],
        case_sensitive: bool = False
    ) -> Optional[str]:
        """
        Match exact trigger from list.
        
        Args:
            text: Normalized text to search in
            triggers: List of exact trigger phrases
            case_sensitive: Whether to match case-sensitively
            
        Returns:
            First matched trigger, or None
        """
        search_text = text if case_sensitive else text.lower()
        
        for trigger in triggers:
            search_trigger = trigger if case_sensitive else trigger.lower()
            if search_trigger in search_text:
                return trigger
        
        return None
    
    def _match_regex_triggers(
        self,
        text: str,
        patterns: List[str]
    ) -> Optional[str]:
        """
        Match triggers using regex patterns.
        
        Useful for capturing inflected forms and variations:
        - Verb conjugations: agendar, agenda, agendarei, agendaremos, etc.
        - Temporal phrases: próximo mês, próxima semana, etc.
        - Formatted dates: DD/MM/YYYY, DD-MM-YYYY, etc.
        
        Args:
            text: Text to search in
            patterns: List of regex patterns
            
        Returns:
            First matched substring, or None
        """
        for pattern in patterns:
            try:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(0)
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
                continue
        
        return None
    
    def _match_lemma_triggers(
        self,
        text: str,
        lemmas_list: List[str]
    ) -> Optional[str]:
        """
        Match triggers based on lemmatization.
        
        Lemmatization handles morphological variations:
        - "agendar", "agenda", "agendando" → all match lemma "agendar"
        - "cancelar", "cancelamento", "cancelada" → all match lemma "cancelar"
        
        Args:
            text: Text to lemmatize and search in
            lemmas_list: Target lemmas to find
            
        Returns:
            First occurrence in original text of a lemma match, or None
        """
        if not self.use_lemmatization or not self.nlp:
            return None
        
        try:
            doc = self.nlp(text[:1000000])
            for token in doc:
                if token.lemma_ in lemmas_list:
                    return token.text
        except Exception as e:
            logger.warning(f"Lemma matching failed: {e}")
        
        return None
    
    def extract_trigger(
        self,
        text: str,
        intent: str
    ) -> Optional[Dict[str, str]]:
        """
        Extract the trigger word/phrase based on intent.
        
        Strategy (in order of priority):
        1. Exact match from intent-specific exact triggers
        2. Regex pattern match from intent-specific patterns
        3. Lemma-based match (if spaCy enabled)
        
        Returns the FIRST valid trigger found, prioritizing high-precision matches.
        
        Args:
            text: Email body text
            intent: Predicted intent class (e.g., "agendamento_reuniao")
            
        Returns:
            Dictionary with keys:
                - 'trigger': The extracted trigger word/phrase
                - 'method': Matching method ('exact', 'regex', 'lemma', or 'none')
                - 'intent': The intent used for extraction
            Or None if no trigger found
            
        Raises:
            ValueError: If intent is not recognized
            
        Example:
            >>> extractor = TriggerExtractor()
            >>> result = extractor.extract_trigger(
            ...     "Vamos agendar uma reunião para amanhã?",
            ...     "agendamento_reuniao"
            ... )
            >>> print(result)
            {
                'trigger': 'agendar',
                'method': 'exact',
                'intent': 'agendamento_reuniao'
            }
        """
        # Validate intent
        try:
            intent_enum = EmailIntent(intent)
        except ValueError:
            valid_intents = [e.value for e in EmailIntent]
            raise ValueError(
                f"Intent '{intent}' not recognized. Valid intents: {valid_intents}"
            )
        
        # Empty text handling
        if not text or not text.strip():
            logger.warning("Empty text provided to extract_trigger")
            return None
        
        normalized_text = self._normalize_text(text)
        lexicon = self.TRIGGER_LEXICONS.get(intent_enum, {})
        
        # Priority 1: Exact match
        if "exact" in lexicon:
            trigger = self._match_exact_trigger(
                normalized_text,
                lexicon["exact"],
                case_sensitive=False
            )
            if trigger:
                logger.debug(f"Extracted trigger (exact): '{trigger}' for intent: {intent}")
                return {
                    "trigger": trigger,
                    "method": "exact",
                    "intent": intent
                }
        
        # Priority 2: Regex pattern match
        if "regex" in lexicon:
            trigger = self._match_regex_triggers(normalized_text, lexicon["regex"])
            if trigger:
                logger.debug(f"Extracted trigger (regex): '{trigger}' for intent: {intent}")
                return {
                    "trigger": trigger,
                    "method": "regex",
                    "intent": intent
                }
        
        # Priority 3: Lemma-based match
        if "lemma" in lexicon and self.use_lemmatization:
            trigger = self._match_lemma_triggers(text, lexicon["lemma"])
            if trigger:
                logger.debug(f"Extracted trigger (lemma): '{trigger}' for intent: {intent}")
                return {
                    "trigger": trigger,
                    "method": "lemma",
                    "intent": intent
                }
        
        logger.debug(f"No trigger found for intent: {intent}")
        return None
    
    def extract_trigger_batch(
        self,
        texts: List[str],
        intents: List[str]
    ) -> List[Optional[Dict[str, str]]]:
        """
        Extract triggers from multiple email texts.
        
        Args:
            texts: List of email body texts
            intents: List of predicted intents (same length as texts)
            
        Returns:
            List of trigger results (or None for each sample)
            
        Raises:
            ValueError: If texts and intents have different lengths
        """
        if len(texts) != len(intents):
            raise ValueError(
                f"texts and intents must have same length: "
                f"{len(texts)} vs {len(intents)}"
            )
        
        results = []
        for text, intent in zip(texts, intents):
            try:
                result = self.extract_trigger(text, intent)
                results.append(result)
            except Exception as e:
                logger.error(f"Error extracting trigger: {e}")
                results.append(None)
        
        return results


# ============================================================================
# EXPLANATION: WHY TRIGGER EXTRACTION DEPENDS ON INTENT
# ============================================================================
"""
1. **Word Sense Ambiguity**: Words have different meanings in different contexts

   Example: "dia" (day)
   - agendamento_reuniao: "Qual dia você está disponível?" → TRIGGER
   - nao_reuniao: "Que dia bonito!" → NOT a trigger
   
   Without intent knowledge, we'd extract false positives.

2. **Density vs. Relevance**: High-frequency words become noise without intent

   The word "reunião" appears in many emails, but it's only a trigger in
   agendamento_reuniao and cancelamento_reuniao, not in REUNIAO_CONFIRMADA.
   
3. **Domain-Specific Vocabulary**: Different intents use different lexicons

   - agendamento_reuniao uses: "agendar", "marcar", "quando", "disponível"
   - cancelamento_reuniao uses: "cancelar", "adiar", "desculpa", "conflito"
   - REUNIAO_CONFIRMADA uses: "confirmar", "consigo", "estar", "estarei"
   
   Same trigger words (like "quando") appear in multiple intents but serve
   different purposes. Intent provides disambiguation.

4. **Efficiency**: Intent-aware extraction is computationally cheaper

   Instead of searching ALL possible triggers across ALL intents,
   we only search the relevant set for the given intent.

5. **Explainability**: Models become auditable and debuggable

   "Why did you extract 'cancelar'?"
   "Because the predicted intent was cancelamento_reuniao."
   
   This is valuable for production systems where decisions must be justified.
"""


# ============================================================================
# LIMITATIONS OF LEXICAL APPROACHES
# ============================================================================
"""
1. **Vocabulary Coverage**: Limited by hand-curated trigger sets

   Problem: New trigger expressions outside the lexicon are missed
   Example: "meet me thursday" (English mixed with Portuguese)
   Solution: Continuously expand lexicons with real data
   
2. **Contextual Misses**: Regex and exact matching don't understand context

   Problem: "Não posso agendar" (I can't schedule) would match as trigger
            for BOTH agendamento_reuniao AND cancelamento_reuniao
   Solution: Require more surrounding context or use sequence models
   
3. **Inflection Handling**: Limited by regex coverage

   Problem: Uncommon verb forms or regional variations not covered
   Example: Brazilian Portuguese "a gente marca" vs European "nós marcamos"
   Solution: Use comprehensive lemmatization or expand regex
   
4. **Negation**: Lexical triggers don't capture negation scope

   Problem: "Não quero cancelar a reunião" = trigger is found but intent is wrong
   Solution: Need syntax awareness (dependency parsing) or transformers
   
5. **Synonym Coverage**: Manual lexicons miss synonyms and related words

   Problem: "reagendar" (reschedule) might not be in agendamento lexicon
   Solution: Use word embeddings or pretrained models
   
6. **Domain Shift**: Lexicons don't adapt to new email domains

   Problem: Technical support emails vs. sales emails use different triggers
   Solution: Domain-adaptive training or transfer learning

7. **Multi-word Expressions**: Hard to capture without linguistic knowledge

   Problem: "colocar na agenda" is a trigger but might be split wrongly
   Solution: Use phrase embeddings or sequence labeling models
"""


# ============================================================================
# EVOLUTION TO TRANSFORMER-BASED TRIGGER DETECTION
# ============================================================================
"""
A transformer-based trigger detector would:

1. **Input Representation**:
   - Use BERT/RoBERTa embeddings for Portuguese text
   - Concatenate with intent embedding as context
   
   input = [CLS] text [SEP] intent [SEP]

2. **Sequence Labeling Approach** (BIO tagging):
   - Label each token as Begin-trigger (B), Inside-trigger (I), or Outside (O)
   - Model learns what constitutes a trigger given the intent
   
   Text: "Vamos agendar uma reunião"
   Tags: O         B      O     O
   
3. **Advantages over Lexical**:
   - Learns trigger patterns from data rather than manual curation
   - Handles context: "Não [posso|quero] agendar" → different predictions
   - Captures synonyms: learns that "marcar" and "agendar" are similar
   - Learns negation: understands "não agendar" means NO trigger
   - Adapts to domain: fine-tune on your specific email domain
   - Handles inflections: learns morphological patterns
   
4. **Implementation with HuggingFace**:

   from transformers import AutoTokenizer, AutoModelForTokenClassification
   
   # Load Portuguese model
   model = AutoModelForTokenClassification.from_pretrained(
       "neuralmind/bert-base-portuguese-cased",
       num_labels=3  # B, I, O
   )
   tokenizer = AutoTokenizer.from_pretrained(
       "neuralmind/bert-base-portuguese-cased"
   )
   
   # Fine-tune on your labeled data
   # Use BIO labels: 0=O, 1=B, 2=I
   
5. **Data Requirements for Fine-tuning**:
   - ~500-1000 labeled examples per intent
   - Annotate trigger spans (starting and ending positions)
   - Example format:
     {
       "text": "Vamos agendar uma reunião",
       "intent": "agendamento_reuniao",
       "trigger": "agendar",
       "trigger_start": 6,  # Character position
       "trigger_end": 13
     }

6. **Evaluation Metrics**:
   - Token-level F1 score
   - Exact match rate (full trigger phrase)
   - Partial match rate (overlapping spans)
   - Per-intent performance

7. **Deployment**:
   - Use ONNX export for production efficiency
   - Cache embeddings for repeated texts
   - Fallback to lexical extraction if transformer fails
   
8. **Hybrid Approach** (Recommended):
   - Use transformer for difficult cases
   - Use lexical for common, unambiguous triggers
   - Ensemble both predictions
   - This is faster AND more accurate
"""


# ============================================================================
# PRODUCTION INTEGRATION EXAMPLE
# ============================================================================
"""
from preprocessing.trigger_extraction import TriggerExtractor
from models.predict_intent import IntentClassifier

# 1. Initialize
intent_classifier = IntentClassifier()
trigger_extractor = TriggerExtractor(use_lemmatization=True)

# 2. Process email
email_text = "Olá, gostaria de agendar uma reunião para próxima semana."

# 3. Classify intent
intent_probs = intent_classifier.predict(email_text)
predicted_intent = max(intent_probs, key=intent_probs.get)

# 4. Extract trigger
trigger_result = trigger_extractor.extract_trigger(email_text, predicted_intent)

# 5. Use results
if trigger_result:
    print(f"Trigger: {trigger_result['trigger']}")
    print(f"Method: {trigger_result['method']}")
else:
    print("No trigger found")

# 6. For monitoring/improvement
if trigger_result is None:
    # Log for manual review and potential lexicon expansion
    log_missed_trigger(email_text, predicted_intent)
"""
