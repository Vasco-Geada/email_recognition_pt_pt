"""
Argument Extraction Module for Portuguese Meeting Emails

This module extracts structured arguments (participants, time, location, topic)
from email bodies using spaCy NER, regex patterns, and lexical heuristics.


Purpose: Baseline for event and temporal expression extraction in Portuguese
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import spacy
from spacy.tokens import Doc


logger = logging.getLogger(__name__)


@dataclass
class ArgumentSpan:
    """Represents an extracted argument with exact text span."""
    text: str                    # Exact text from email
    span_start: int              # Character offset start
    span_end: int                # Character offset end
    confidence: float = 1.0      # Extraction confidence (0-1)
    extraction_method: str = ""  # Method used: 'ner', 'regex', 'heuristic'
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExtractedArguments:
    """Complete set of extracted arguments from an email."""
    participants: List[ArgumentSpan] = field(default_factory=list)
    time_expressions: List[ArgumentSpan] = field(default_factory=list)
    locations: List[ArgumentSpan] = field(default_factory=list)
    topics: List[ArgumentSpan] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with serialized spans."""
        return {
            'participants': [s.to_dict() for s in self.participants],
            'time_expressions': [s.to_dict() for s in self.time_expressions],
            'locations': [s.to_dict() for s in self.locations],
            'topics': [s.to_dict() for s in self.topics],
        }
    
    def summary(self) -> Dict:
        """Return summary with just text values."""
        return {
            'participants': [s.text for s in self.participants],
            'time_expressions': [s.text for s in self.time_expressions],
            'locations': [s.text for s in self.locations],
            'topics': [s.text for s in self.topics],
        }


class TemporalExpressionExtractor:
    """
    Extracts Portuguese temporal expressions using pattern-based regex matching.
    
    Handles:
    - Specific dates: "5 de março"
    - Weekdays: "segunda-feira", "seg."
    - Relative expressions: "amanhã", "próxima semana"
    - Time expressions: "às 15h", "15:00"
    - Informal expressions: "depois de almoço", "de manhã"
    """
    
    # Portuguese weekday patterns
    WEEKDAYS = {
        r'\b(?:segunda|seg)\.?\s*-?\s*(?:feira)?\b': 'monday',
        r'\b(?:terça|ter)\.?\s*-?\s*(?:feira)?\b': 'tuesday',
        r'\b(?:quarta|qua)\.?\s*-?\s*(?:feira)?\b': 'wednesday',
        r'\b(?:quinta|qui)\.?\s*-?\s*(?:feira)?\b': 'thursday',
        r'\b(?:sexta|sex)\.?\s*-?\s*(?:feira)?\b': 'friday',
        r'\b(?:sábado|sab)\.?\b': 'saturday',
        r'\b(?:domingo|dom)\.?\b': 'sunday',
    }
    
    # Relative temporal expressions
    RELATIVE_PATTERNS = [
        r'\b(?:amanhã|manha|amanhã)\b',           # tomorrow
        r'\b(?:hoje|hj)\b',                        # today
        r'\b(?:ontem|ont)\b',                      # yesterday
        r'\b(?:próxima|proxima)\s+(?:semana|segunda)',  # next week/monday
        r'\b(?:próximo|proximo)\s+(?:mês|mes|ano)\b',   # next month/year
        r'\b(?:esta|este)\s+(?:semana|segunda|sexta)\b', # this week/monday/friday
        r'\b(?:semana)\s+(?:que|k)\s+(?:vem|vém)\b',    # week that comes
        r'\b(?:daqui)\s+a\s+(?:\d+)\s+(?:dias|semanas|horas)\b', # in X days/weeks
    ]
    
    # Informal time expressions
    INFORMAL_TIME = [
        r'\b(?:de|por)\s+(?:manhã|tarde|noite)\b',    # morning/afternoon/evening
        r'\b(?:depois|ap[ó|o]s)\s+(?:de\s+)?(?:almoço|café|trabalho)\b',  # after
        r'\b(?:antes|ac)\s+(?:de\s+)?(?:almoço|café)\b',  # before
        r'\b(?:ao\s+)?(?:nível|final)\s+(?:de\s+)?(?:semana|dia)\b',  # end of week/day
    ]
    
    # Time of day patterns
    TIME_PATTERNS = [
        r'\b(?:às|as|a)\s+(?:\d{1,2}):?(?:\d{2})?\s*(?:h|horas)?\b',  # às 15h, às 15:00
        r'\b\d{1,2}:?\d{2}\s*(?:h|horas)?\b',  # 15:00, 15h
        r'\b(?:\d{1,2})\s*horas?\b',           # 15 horas
    ]
    
    # Full date patterns
    DATE_PATTERNS = [
        r'\b\d{1,2}\s+de\s+(?:janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro|jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)(?:\s+de\s+\d{2,4})?\b',  # 5 de março
        r'\b(?:jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)\s+\d{1,2}(?:\s+\d{4})?\b',  # mar 05 2024
        r'\b\d{1,2}[-/]\d{1,2}(?:[-/]\d{2,4})?\b',  # 05-03, 05/03/2024
    ]
    
    def __init__(self):
        """Initialize temporal expression extractor."""
        self.weekday_regex = [re.compile(p, re.IGNORECASE) for p in self.WEEKDAYS.keys()]
        self.relative_regex = [re.compile(p, re.IGNORECASE) for p in self.RELATIVE_PATTERNS]
        self.informal_regex = [re.compile(p, re.IGNORECASE) for p in self.INFORMAL_TIME]
        self.time_regex = [re.compile(p, re.IGNORECASE) for p in self.TIME_PATTERNS]
        self.date_regex = [re.compile(p, re.IGNORECASE) for p in self.DATE_PATTERNS]
    
    def extract(self, text: str) -> List[ArgumentSpan]:
        """
        Extract all temporal expressions from text.
        
        Args:
            text: Email body text
            
        Returns:
            List of ArgumentSpan objects with temporal expressions
        """
        spans = []
        
        # Check each pattern type
        for regex_list, method in [
            (self.date_regex, 'date'),
            (self.time_regex, 'time'),
            (self.weekday_regex, 'weekday'),
            (self.relative_regex, 'relative'),
            (self.informal_regex, 'informal'),
        ]:
            for regex in regex_list:
                for match in regex.finditer(text):
                    span = ArgumentSpan(
                        text=match.group(),
                        span_start=match.start(),
                        span_end=match.end(),
                        confidence=0.9,  # Pattern-based have high confidence
                        extraction_method=f'regex_{method}',
                    )
                    spans.append(span)
        
        # Remove duplicates and overlaps, keeping longest matches
        return self._deduplicate_spans(spans)
    
    @staticmethod
    def _deduplicate_spans(spans: List[ArgumentSpan]) -> List[ArgumentSpan]:
        """Remove overlapping spans, keeping longest matches."""
        if not spans:
            return []
        
        # Sort by start position, then by length (descending)
        sorted_spans = sorted(spans, key=lambda s: (s.span_start, -(s.span_end - s.span_start)))
        
        deduplicated = []
        for span in sorted_spans:
            # Check if this span overlaps with any already added
            has_overlap = False
            for existing in deduplicated:
                if not (span.span_end <= existing.span_start or span.span_start >= existing.span_end):
                    has_overlap = True
                    break
            if not has_overlap:
                deduplicated.append(span)
        
        return sorted(deduplicated, key=lambda s: s.span_start)


class LocationExtractor:
    """
    Extracts room/location information using regex patterns and heuristics.
    
    Handles:
    - Room numbers: "sala 203", "escritório 5"
    - Building references: "1º andar", "piso 2"
    - Named locations: "Auditório A", "Lab de IA"
    - Email mentions implying places
    """
    
    LOCATION_PATTERNS = [
        # Room patterns
        r'\b(?:sala|salas|escritório|escritorios|gabinete|gabinetes|auditório|auditórios)\s+(?:de\s+)?(?:n°|número|nº|#)?(\d+|[A-Z])\b',
        # Floor patterns
        r'\b(?:\d+)[°º]\s*(?:andar|piso)\b',
        r'\b(?:andar|piso)\s+(?:\d+|térreo|terreo|cave)\b',
        # Named location patterns
        r'\b(?:auditório|auditorio|laboratório|laboratorio|lab|anfiteatro|anfiteatro|biblioteca|cafetaria|refeitório|refeitorio|cantina|armazém|armazem)\b',
        # Building/campus references
        r'\b(?:bloco|edifício|edificio|campus|sede|filial)\s+[A-Z0-9]+\b',
        # Street/address patterns (basic)
        r'\b(?:rua|avenida|av|alameda|praça|pç)\s+[A-Z][a-záéíóúàâêõç\s]+\b',
    ]
    
    def __init__(self):
        """Initialize location extractor."""
        self.location_regex = [re.compile(p, re.IGNORECASE) for p in self.LOCATION_PATTERNS]
    
    def extract(self, text: str) -> List[ArgumentSpan]:
        """
        Extract location references from text.
        
        Args:
            text: Email body text
            
        Returns:
            List of ArgumentSpan objects with locations
        """
        spans = []
        
        for regex in self.location_regex:
            for match in regex.finditer(text):
                span = ArgumentSpan(
                    text=match.group(),
                    span_start=match.start(),
                    span_end=match.end(),
                    confidence=0.85,
                    extraction_method='regex_location',
                )
                spans.append(span)
        
        return self._deduplicate_spans(spans)
    
    @staticmethod
    def _deduplicate_spans(spans: List[ArgumentSpan]) -> List[ArgumentSpan]:
        """Remove overlapping spans, keeping longest matches."""
        if not spans:
            return []
        
        sorted_spans = sorted(spans, key=lambda s: (s.span_start, -(s.span_end - s.span_start)))
        deduplicated = []
        
        for span in sorted_spans:
            has_overlap = False
            for existing in deduplicated:
                if not (span.span_end <= existing.span_start or span.span_start >= existing.span_end):
                    has_overlap = True
                    break
            if not has_overlap:
                deduplicated.append(span)
        
        return sorted(deduplicated, key=lambda s: s.span_start)


class ParticipantExtractor:
    """
    Extracts participant names using spaCy NER and email-specific heuristics.
    
    Uses:
    - spaCy PER entities (PERSON)
    - Email address extraction
    - Common patterns like "com o João", "entre X e Y"
    """
    
    def __init__(self, nlp_model):
        """
        Initialize participant extractor.
        
        Args:
            nlp_model: Loaded spaCy model (pt_core_news_sm or similar)
        """
        self.nlp = nlp_model
    
    # Patterns for extracting participants
    PARTICIPANT_PATTERNS = [
        r'(?:com|entre|por|de)\s+(?:o\s+|a\s+|os\s+|as\s+)?([A-Z][a-záéíóúàâêõç\s]+?)(?:\s*(?:de|do|da|e|,|$))',  # com o [Name]
        r'(?:participar|presença|estar|comparecer|assistir)\s+(?:à|a|do|da)\s+([A-Z][a-záéíóúàâêõç\s]+?)(?:\s*(?:,|$))',  # participar à [Name]
    ]
    
    # Email pattern
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    def extract(self, text: str, doc: Optional[Doc] = None) -> List[ArgumentSpan]:
        """
        Extract participants from text.
        
        Args:
            text: Email body text
            doc: Optional pre-processed spaCy Doc object
            
        Returns:
            List of ArgumentSpan objects with participant names
        """
        spans = []
        
        # Use spaCy NER if doc provided
        if doc is None:
            doc = self.nlp(text)
        
        # Extract PERSON entities
        for ent in doc.ents:
            if ent.label_ == 'PER':
                span = ArgumentSpan(
                    text=ent.text,
                    span_start=ent.start_char,
                    span_end=ent.end_char,
                    confidence=0.8,
                    extraction_method='ner_spacy',
                )
                spans.append(span)
        
        # Extract email addresses
        email_regex = re.compile(self.EMAIL_PATTERN)
        for match in email_regex.finditer(text):
            span = ArgumentSpan(
                text=match.group(),
                span_start=match.start(),
                span_end=match.end(),
                confidence=0.95,
                extraction_method='regex_email',
            )
            spans.append(span)
        
        # Try pattern-based extraction for informal contexts
        for pattern in self.PARTICIPANT_PATTERNS:
            regex = re.compile(pattern, re.IGNORECASE)
            for match in regex.finditer(text):
                # Take the group that actually captured the name
                for group_idx in range(1, len(match.groups()) + 1):
                    if match.group(group_idx):
                        # Get actual position in text
                        name_text = match.group(group_idx).strip()
                        # Find it in original match to get correct span
                        name_pos = match.start() + match.group(0).find(name_text)
                        span = ArgumentSpan(
                            text=name_text,
                            span_start=name_pos,
                            span_end=name_pos + len(name_text),
                            confidence=0.6,
                            extraction_method='regex_heuristic',
                        )
                        spans.append(span)
                        break
        
        return self._deduplicate_spans(spans)
    
    @staticmethod
    def _deduplicate_spans(spans: List[ArgumentSpan]) -> List[ArgumentSpan]:
        """Remove duplicates, preferring higher confidence methods."""
        if not spans:
            return []
        
        # Group by text (case-insensitive)
        unique_by_text = {}
        for span in spans:
            key = span.text.lower()
            if key not in unique_by_text or span.confidence > unique_by_text[key].confidence:
                unique_by_text[key] = span
        
        return sorted(unique_by_text.values(), key=lambda s: s.span_start)


class TopicExtractor:
    """
    Extracts meeting topics using keyword heuristics and textual patterns.
    
    Strategy:
    - Keyword/n-gram frequency analysis (excluding stop words)
    - Sentence noun phrase extraction using spaCy
    - Context from intent and subject line
    - Trigger words indicating topic relevance
    """
    
    # Stop words in Portuguese
    STOP_WORDS = {
        'de', 'do', 'da', 'dos', 'das', 'e', 'ou', 'um', 'uma', 'uns', 'umas',
        'o', 'a', 'os', 'as', 'em', 'em', 'para', 'por', 'com', 'sem', 'sob',
        'sobre', 'que', 'qual', 'quais', 'a', 'à', 'ao', 'aos', 'é', 'são',
        'está', 'estão', 'ser', 'estar', 'ter', 'teve', 'temos', 'tenho',
        'reunião', 'reuniões', 'meeting', 'meetings', 'email', 'emails',
    }
    
    # Topic keywords (indicate relevant content)
    TOPIC_KEYWORDS = {
        'projeto': ['projeto', 'projeto', 'desenvolvimento', 'projeto', 'projeto'],
        'orçamento': ['orçamento', 'orçamento', 'preço', 'custo', 'valor', 'financeiro'],
        'recursos': ['recursos', 'recursos', 'equipa', 'equipe', 'pessoal', 'staff'],
        'cronograma': ['cronograma', 'timeline', 'prazos', 'prazo', 'datas', 'milestones'],
        'qualidade': ['qualidade', 'qc', 'testes', 'testes', 'testes', 'qa', 'qa'],
        'apresentação': ['apresentação', 'apresentação', 'demo', 'demonstração', 'showcace'],
    }
    
    def __init__(self, nlp_model):
        """
        Initialize topic extractor.
        
        Args:
            nlp_model: Loaded spaCy model
        """
        self.nlp = nlp_model
    
    def extract(self, text: str, subject: str = "", intent: str = "") -> List[ArgumentSpan]:
        """
        Extract main topics from email.
        
        Args:
            text: Email body text
            subject: Email subject line
            intent: Predicted email intent (helps contextualize topic)
            
        Returns:
            List of ArgumentSpan objects with topics
        """
        combined_text = f"{subject}. {text}"
        doc = self.nlp(combined_text)
        
        spans = []
        
        # Strategy 1: Extract noun phrases (multi-word expressions)
        noun_phrases = self._extract_noun_phrases(doc)
        spans.extend(noun_phrases)
        
        # Strategy 2: Extract topic keywords
        keyword_spans = self._extract_keywords(text, combined_text)
        spans.extend(keyword_spans)
        
        # Remove duplicates and low-confidence items
        spans = self._deduplicate_and_rank(spans)
        
        # Keep only top topics (limit to 3-5)
        return sorted(spans, key=lambda s: s.confidence, reverse=True)[:5]
    
    def _extract_noun_phrases(self, doc: Doc) -> List[ArgumentSpan]:
        """Extract noun chunks (noun phrases) from spaCy doc."""
        spans = []
        
        for chunk in doc.noun_chunks:
            # Skip if mostly stop words
            tokens = chunk.text.lower().split()
            non_stop = [t for t in tokens if t not in self.STOP_WORDS]
            
            if len(non_stop) > 0 and len(chunk.text) > 2:  # Skip very short phrases
                confidence = len(non_stop) / len(tokens)  # More content = higher confidence
                span = ArgumentSpan(
                    text=chunk.text.lower(),
                    span_start=chunk.start_char,
                    span_end=chunk.end_char,
                    confidence=min(0.9, 0.5 + confidence * 0.4),  # 0.5-0.9
                    extraction_method='spacy_noun_chunks',
                )
                spans.append(span)
        
        return spans
    
    def _extract_keywords(self, text: str, combined_text: str) -> List[ArgumentSpan]:
        """Extract predefined topic keywords."""
        spans = []
        text_lower = text.lower()
        
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            for keyword in keywords:
                regex = re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE)
                for match in regex.finditer(text_lower):
                    span = ArgumentSpan(
                        text=topic,
                        span_start=match.start(),
                        span_end=match.end(),
                        confidence=0.7,
                        extraction_method='keyword_heuristic',
                    )
                    spans.append(span)
        
        return spans
    
    @staticmethod
    def _deduplicate_and_rank(spans: List[ArgumentSpan]) -> List[ArgumentSpan]:
        """Remove duplicates and rank by confidence."""
        if not spans:
            return []
        
        unique_by_text = {}
        for span in spans:
            key = span.text.lower()
            if key not in unique_by_text:
                unique_by_text[key] = span
            else:
                # Keep highest confidence
                if span.confidence > unique_by_text[key].confidence:
                    unique_by_text[key] = span
        
        return sorted(unique_by_text.values(), key=lambda s: s.confidence, reverse=True)


class ArgumentExtractor:
    """
    Main class for extracting structured arguments from Portuguese meeting emails.
    
    Input:
        - email_body: Raw email text
        - email_subject: Email subject line
        - predicted_intent: Output from intent classifier
        - trigger: Optional extracted trigger phrase
    
    Output:
        - ExtractedArguments object with participants, time, location, topic
    """
    
    def __init__(self, model_name: str = "pt_core_news_sm"):
        """
        Initialize argument extractor with spaCy model.
        
        Args:
            model_name: Name of spaCy Portuguese model to load
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.error(
                f"Model {model_name} not found. Install with:\n"
                f"python -m spacy download {model_name}"
            )
            raise
        
        # Initialize component extractors
        self.temporal_extractor = TemporalExpressionExtractor()
        self.location_extractor = LocationExtractor()
        self.participant_extractor = ParticipantExtractor(self.nlp)
        self.topic_extractor = TopicExtractor(self.nlp)
    
    def extract(
        self,
        email_body: str,
        email_subject: str = "",
        predicted_intent: str = "",
        trigger: str = "",
    ) -> ExtractedArguments:
        """
        Extract all arguments from an email.
        
        Args:
            email_body: Email body text
            email_subject: Email subject line
            predicted_intent: Intent classification result
            trigger: Optional trigger word/phrase
            
        Returns:
            ExtractedArguments object with all extracted arguments
        """
        # Preprocess with spaCy
        doc = self.nlp(email_body)
        
        # Extract each argument type
        participants = self.participant_extractor.extract(email_body, doc)
        time_expressions = self.temporal_extractor.extract(email_body)
        locations = self.location_extractor.extract(email_body)
        topics = self.topic_extractor.extract(email_body, email_subject, predicted_intent)
        
        return ExtractedArguments(
            participants=participants,
            time_expressions=time_expressions,
            locations=locations,
            topics=topics,
        )
    
    def extract_with_context(
        self,
        email_body: str,
        email_subject: str = "",
        predicted_intent: str = "",
        trigger: str = "",
        include_confidence: bool = True,
    ) -> Dict:
        """
        Extract arguments and return with full context.
        
        Args:
            email_body: Email body text
            email_subject: Email subject line
            predicted_intent: Intent classification result
            trigger: Optional trigger phrase
            include_confidence: Whether to include confidence scores
            
        Returns:
            Dictionary with extracted arguments and metadata
        """
        arguments = self.extract(email_body, email_subject, predicted_intent, trigger)
        
        result = {
            'extracted_arguments': arguments.to_dict() if include_confidence else arguments.summary(),
            'metadata': {
                'email_subject': email_subject,
                'predicted_intent': predicted_intent,
                'trigger': trigger,
                'extraction_timestamp': datetime.now().isoformat(),
            }
        }
        
        return result
