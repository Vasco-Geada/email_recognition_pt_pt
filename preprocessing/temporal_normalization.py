"""
Temporal Expression Normalization Module for European Portuguese

This module normalizes temporal expressions extracted from Portuguese emails
into structured datetime objects. It supports:
- Relative expressions (amanhã, hoje, para a semana)
- Weekday expressions (segunda, terça, sexta)
- Time expressions (às 15h, 10:30)
- Informal temporal markers (depois de almoço)
- Complex combined expressions


Date: 2026
"""

import re
import logging
import unicodedata
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, time as time_obj
from enum import Enum

logger = logging.getLogger(__name__)


def parse_datetime_value(value: object) -> Optional[datetime]:
    """Parse ISO 8601 or standard RFC 2822 email date metadata."""
    if isinstance(value, datetime):
        return value
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass

    try:
        return parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        return None


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class TemporalType(Enum):
    """Classification of normalized temporal expressions."""
    DATE = "date"              # Specific date (e.g., "segunda-feira, 16 de Abril")
    TIME = "time"              # Specific time only (e.g., "às 14h")
    DATETIME = "datetime"      # Date and time (e.g., "sexta às 15h")
    INTERVAL = "interval"      # Time range (e.g., "depois de almoço")
    RELATIVE = "relative"      # Relative to reference (e.g., "amanhã")
    DURATION = "duration"      # Time span (e.g., "1 hora", "2 semanas")
    UNKNOWN = "unknown"        # Could not determine


@dataclass
class NormalizedTemporal:
    """Structured output for normalized temporal expressions."""
    
    # Input information
    original_text: str                          # Original expression from email
    reference_datetime: datetime                # Reference point for normalization
    
    # Normalization results
    temporal_type: TemporalType                 # Classification
    normalized_datetime: Optional[datetime] = None     # Primary datetime (ISO format)
    normalized_date: Optional[str] = None              # ISO date string (YYYY-MM-DD)
    normalized_time: Optional[str] = None              # ISO time string (HH:MM:SS)
    normalized_datetime_str: Optional[str] = None      # ISO full datetime
    
    # Interval support (for ranges)
    interval_start: Optional[datetime] = None  # Start of interval
    interval_end: Optional[datetime] = None    # End of interval
    interval_start_str: Optional[str] = None   # ISO format
    interval_end_str: Optional[str] = None     # ISO format
    
    # Metadata
    precision: str = "unknown"                 # day/time/exact
    confidence: float = 1.0                    # Confidence score (0-1)
    extraction_method: str = "rule_based"      # Method used
    notes: List[str] = field(default_factory=list)  # Processing notes/warnings

    def canonical_value(self) -> Optional[str]:
        """Return the most precise ISO-compatible value without inventing data."""
        if self.interval_start_str and self.interval_end_str:
            return f"{self.interval_start_str}/{self.interval_end_str}"
        if self.normalized_datetime_str:
            return self.normalized_datetime_str
        if self.normalized_date:
            return self.normalized_date
        if self.normalized_time:
            return self.normalized_time
        return None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with ISO string dates."""
        result = {
            'original_text': self.original_text,
            'reference_datetime': self.reference_datetime.isoformat(),
            'temporal_type': self.temporal_type.value,
            'normalized_datetime': self.normalized_datetime_str,
            'normalized_date': self.normalized_date,
            'normalized_time': self.normalized_time,
            'canonical_value': self.canonical_value(),
            'precision': self.precision,
            'confidence': self.confidence,
            'extraction_method': self.extraction_method,
            'notes': self.notes,
        }
        
        # Add interval if present
        if self.temporal_type == TemporalType.INTERVAL:
            result['interval_start'] = self.interval_start_str
            result['interval_end'] = self.interval_end_str
        
        return result


# ============================================================================
# PORTUGUESE TEMPORAL LEXICONS
# ============================================================================

# Relative expressions (relative to reference date)
RELATIVE_EXPRESSIONS = {
    # Today
    'hoje': 0,
    'agora': 0,
    'daqui a um bocado': 0,
    'depois da daily': 0,
    'depois desta reunião': 0,
    
    # Tomorrow/next day
    'amanhã': 1,
    'no dia a seguir': 1,
    
    # Day after tomorrow
    'depois de amanhã': 2,
    'depois amanhã': 2,
    
    # Yesterday
    'ontem': -1,
    
    # This week
    'esta semana': 'this_week',
    'para a semana': 'next_week',
    'para esta semana': 'this_week',
    
    # This/next month
    'este mês': 'this_month',
    'próximo mês': 'next_month',
    
    # Generic "soon"
    'em breve': 'soon',
    'brevemente': 'soon',
}

# Weekday names
WEEKDAYS = {
    'segunda': 0,           # Monday
    'segunda-feira': 0,
    'terca': 1,             # Tuesday
    'terça': 1,             # Tuesday
    'terça-feira': 1,
    'quarta': 2,            # Wednesday
    'quarta-feira': 2,
    'quinta': 3,            # Thursday
    'quinta-feira': 3,
    'sexta': 4,             # Friday
    'sexta-feira': 4,
    'sábado': 5,            # Saturday
    'sabado': 5,
    'domingo': 6,           # Sunday
}

# Qualifiers for weekdays (próxima = next)
WEEKDAY_QUALIFIERS = {
    'próxima': 'next',
    'proxima': 'next',
    'próximo': 'next',
    'proximo': 'next',
    'esta': 'this',
    'este': 'this',
}

# Time of day approximations
TIME_OF_DAY = {
    # Morning
    'manhã': (6, 0, 12, 0),            # 6h-12h
    'manha': (6, 0, 12, 0),            # 6h-12h
    'de manhã': (6, 0, 12, 0),
    'de manha': (6, 0, 12, 0),
    'pela manhã': (6, 0, 12, 0),
    'pela manha': (6, 0, 12, 0),
    'antes de almoço': (6, 0, 12, 0),
    'antes de almoco': (6, 0, 12, 0),
    
    # Afternoon
    'tarde': (12, 0, 18, 0),           # 12h-18h
    'de tarde': (12, 0, 18, 0),
    'pela tarde': (12, 0, 18, 0),
    'antes de sair': (12, 0, 18, 0),
    'antes de saires': (12, 0, 18, 0),
    'antes de sairmos': (12, 0, 18, 0),
    'antes ires para casa': (12, 0, 18, 0),
    'antes ir para casa': (12, 0, 18, 0),
    'antes do dia terminar': (12, 0, 18, 0),
    
    # Evening
    'noite': (18, 0, 23, 59),          # 18h-23:59h
    'de noite': (18, 0, 23, 59),
    'pela noite': (18, 0, 23, 59),
    
    # After lunch
    'depois de almoço': (14, 00, 15, 30),  # ~14:00-15:30
    'pós almoço': (14, 00, 15, 30),
    
    # After breakfast
    'depois do café': (9, 0, 10, 30),  # ~9:00-10:30
    'pós café': (9, 0, 10, 30),
    'depois do pequeno almoço': (9, 0, 10, 30),
    'depois do pequeno-almoço': (9, 0, 10, 30),
    'assim que entrarmos': (9, 0, 10, 30),
}

# Month names
MONTHS = {
    'janeiro': 1, 'jan': 1,
    'fevereiro': 2, 'fev': 2,
    'março': 3, 'marco': 3, 'mar': 3,
    'abril': 4, 'abr': 4,
    'maio': 5, 'mai': 5,
    'junho': 6, 'jun': 6,
    'julho': 7, 'jul': 7,
    'agosto': 8, 'ago': 8,
    'setembro': 9, 'set': 9,
    'outubro': 10, 'out': 10,
    'novembro': 11, 'nov': 11,
    'dezembro': 12, 'dez': 12,
}

RELATIVE_EXPRESSIONS.update({
    "amanha": 1,
    "depois de amanha": 2,
    "depois amanha": 2,
    "na proxima semana": "next_week",
    "proxima semana": "next_week",
    "este mes": "this_month",
    "proximo mes": "next_month",
    "final desta semana": "end_of_week",
    "final da semana": "end_of_week",
    "fim desta semana": "end_of_week",
    "fim da semana": "end_of_week",
})


# ============================================================================
# REGEX PATTERNS
# ============================================================================

class TemporalPatterns:
    """Compiled regex patterns for temporal expression matching."""
    
    # Time patterns - more specific to avoid false positives
    # Examples: "15h", "15:30", "15h30", "às 15h", "às 15:30"
    # Must have explicit time separator (h, :, or .) to avoid matching dates
    TIME_PATTERN = re.compile(
        r'(?:(?:as?|pelas?)\s*)?'
        r'(\d{1,2})(?:h(?:(\d{2}))?|[:\.](\d{1,2})|\s*horas?)'
        r'(?:\s*min)?(?!\s*de)',
        re.IGNORECASE
    )
    
    # Explicit dates with either a month name or an unambiguous numeric separator.
    # Examples: "16 de Abril", "16 de abril de 2026", "16/04/2026", "16-04-2026"
    EXPLICIT_DATE_PATTERN = re.compile(
        r'\b(?:'
        r'(\d{1,2})\s+de\s+([a-z]+)(?:\s+de\s+(\d{4}))?'
        r'|'
        r'(\d{1,2})\s*([/-])\s*(\d{1,2})(?:\s*[/-]\s*(\d{2,4}))?'
        r')\b',
        re.IGNORECASE,
    )

    # Day of month without an explicit month: "dia 18 às 9h30".
    DAY_OF_MONTH_PATTERN = re.compile(
        r'\bdia\s+(\d{1,2})\b',
        re.IGNORECASE,
    )
    
    # Weekday patterns
    # Examples: "segunda", "próxima sexta", "esta terça"
    WEEKDAY_PATTERN = re.compile(
        r'(?:(proxima|proximo|esta|este)\s+)?'
        r'(segunda(?:-feira)?|terca(?:-feira)?|quarta(?:-feira)?|'
        r'quinta(?:-feira)?|sexta(?:-feira)?|sabado|domingo)',
        re.IGNORECASE
    )
    
    # Relative expression patterns (amanhã, hoje, ontem, etc.)
    RELATIVE_PATTERN = re.compile(
        r'(depois de amanha|depois amanha|amanha|hoje|ontem|agora|'
        r'para a semana|esta semana|na proxima semana|proxima semana|'
        r'proximo mes|este mes|em breve|'
        r'final desta semana|final da semana|fim desta semana|fim da semana)',
        re.IGNORECASE
    )
    
    # Time of day patterns (manhã, tarde, noite, depois de almoço)
    TIME_OF_DAY_PATTERN = re.compile(
        r'\b(durante\s+a\s+manha|ao\s+final\s+do\s+dia|final\s+do\s+dia|'
        r'pela\s+manha|de\s+manha|manha|pela\s+tarde|de\s+tarde|tarde|'
        r'pela\s+noite|de\s+noite|noite|apos\s+almoco|'
        r'depois\s+de\s+almoco|pos\s+almoco|apos\s+cafe|'
        r'depois\s+do\s+cafe|pos\s+cafe)\b',
        re.IGNORECASE
    )
    
    # Relative day offsets
    # Examples: "em 3 dias", "daqui a 2 dias"
    RELATIVE_OFFSET_PATTERN = re.compile(
        r'(?:em|daqui\s+a|dentro\s+de)\s+(\d+)\s+(dias?|semanas?|horas?)',
        re.IGNORECASE
    )


# ============================================================================
# TEMPORAL NORMALIZER
# ============================================================================

class TemporalNormalizer:
    """
    Main class for normalizing temporal expressions in Portuguese.
    
    Strategy:
    1. Tokenize and clean the input expression
    2. Match against regex patterns (with priority ordering)
    3. Apply rule-based normalization for each matched pattern
    4. Handle ambiguity deterministically (pick highest confidence parsing)
    5. Return structured NormalizedTemporal object
    """
    
    def __init__(self):
        """Initialize patterns and lexicons."""
        self.patterns = TemporalPatterns()
        self.weekdays = WEEKDAYS
        self.months = MONTHS
        self.relative_expr = RELATIVE_EXPRESSIONS
        self.time_of_day = TIME_OF_DAY

    @staticmethod
    def _normalize_text(value: str) -> str:
        """Normalize accents and spacing while keeping offsets irrelevant here."""
        decomposed = unicodedata.normalize("NFKD", str(value or ""))
        without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
        return re.sub(r"\s+", " ", without_accents).strip().lower()

    @staticmethod
    def _datetime_for_date(reference: datetime, year: int, month: int, day: int) -> datetime:
        return datetime(year, month, day, tzinfo=reference.tzinfo)

    @staticmethod
    def _set_date(
        result: NormalizedTemporal,
        value: datetime,
        temporal_type: TemporalType = TemporalType.DATE,
        precision: str = "day",
    ) -> None:
        result.normalized_datetime = None
        result.normalized_datetime_str = None
        result.normalized_date = value.strftime("%Y-%m-%d")
        result.normalized_time = None
        result.temporal_type = temporal_type
        result.precision = precision

    @staticmethod
    def _set_datetime(
        result: NormalizedTemporal,
        value: datetime,
        precision: str = "exact",
        temporal_type: TemporalType = TemporalType.DATETIME,
    ) -> None:
        value = value.replace(second=0, microsecond=0)
        result.normalized_datetime = value
        result.normalized_datetime_str = value.isoformat()
        result.normalized_date = value.strftime("%Y-%m-%d")
        result.normalized_time = value.strftime("%H:%M:%S")
        result.temporal_type = temporal_type
        result.precision = precision

    @staticmethod
    def _set_interval(
        result: NormalizedTemporal,
        start: datetime,
        end: datetime,
        precision: str,
    ) -> None:
        result.normalized_datetime = None
        result.normalized_datetime_str = None
        result.normalized_date = start.strftime("%Y-%m-%d")
        result.normalized_time = None
        result.interval_start = start
        result.interval_end = end
        result.interval_start_str = start.isoformat()
        result.interval_end_str = end.isoformat()
        result.temporal_type = TemporalType.INTERVAL
        result.precision = precision

    def _apply_time_of_day_interval(
        self,
        result: NormalizedTemporal,
        expr: str,
    ) -> bool:
        match = self.patterns.TIME_OF_DAY_PATTERN.search(expr)
        if not match or not result.normalized_date:
            return False

        time_range = self._get_time_of_day_range(match.group(0))
        if not time_range:
            return False

        start_h, start_m, end_h, end_m = time_range
        base_date = datetime.fromisoformat(result.normalized_date)
        start = base_date.replace(
            hour=start_h,
            minute=start_m,
            tzinfo=result.reference_datetime.tzinfo,
        )
        end = base_date.replace(
            hour=end_h,
            minute=end_m,
            tzinfo=result.reference_datetime.tzinfo,
        )
        self._set_interval(result, start, end, "time_of_day")
        result.notes.append(f"Applied time-of-day interval: '{match.group(0)}'")
        return True
    
    def normalize(
        self,
        temporal_expression: str,
        reference_datetime: datetime = None,
        context: Optional[Dict] = None,
    ) -> NormalizedTemporal:
        """
        Normalize a temporal expression to a structured datetime.
        
        Args:
            temporal_expression: The text to normalize (e.g., "sexta às 15h")
            reference_datetime: Reference point for relative expressions (default: now)
            context: Optional additional context (email intent, previous expressions, etc.)
        
        Returns:
            NormalizedTemporal object with normalized datetime(s)
        """
        
        # Use current datetime as reference if not provided
        if reference_datetime is None:
            reference_datetime = datetime.now()
        
        # Normalize accents because project datasets contain accented and ASCII PT-PT.
        expr_normalized = self._normalize_text(temporal_expression)
        
        # Initialize result
        result = NormalizedTemporal(
            original_text=temporal_expression,
            reference_datetime=reference_datetime,
            temporal_type=TemporalType.UNKNOWN,
        )
        
        try:
            # Try parsing strategies in priority order
            
            # 1. Explicit date patterns (highest priority - most specific)
            parsed = self._parse_explicit_date(expr_normalized, result)
            if parsed:
                return result

            # 2. Day of month without month name ("dia 18 às 9h30")
            parsed = self._parse_day_of_month(expr_normalized, result)
            if parsed:
                return result
            
            # 3. Complex patterns (combinations)
            parsed = self._parse_complex_expression(expr_normalized, result)
            if parsed:
                return result

            # 4. Relative numeric offsets ("daqui a 2 dias")
            parsed = self._parse_relative_offset(expr_normalized, result)
            if parsed:
                return result

            # 5. Weekday patterns
            parsed = self._parse_weekday(expr_normalized, result)
            if parsed:
                return result
            
            # 6. Relative expressions
            parsed = self._parse_relative_expression(expr_normalized, result)
            if parsed:
                return result
            
            # 7. Time of day standalone patterns
            parsed = self._parse_time_of_day_standalone(expr_normalized, result)
            if parsed:
                return result
            
            # 8. Time only patterns
            parsed = self._parse_time_only(expr_normalized, result)
            if parsed:
                return result
            
            # If nothing matched, set as unknown
            result.notes.append(f"Could not parse temporal expression: '{temporal_expression}'")
            result.confidence = 0.0
            
        except Exception as e:
            result.notes.append(f"Error during normalization: {str(e)}")
            result.confidence = 0.0
            logger.error(f"Error normalizing temporal expression '{temporal_expression}': {e}")
        
        return result
    
    def _parse_explicit_date(
        self,
        expr: str,
        result: NormalizedTemporal
    ) -> bool:
        """
        Parse explicit date patterns (e.g., "16 de Abril", "16/04/2026").
        
        Returns True if successfully parsed.
        """
        match = self.patterns.EXPLICIT_DATE_PATTERN.search(expr)
        if not match:
            return False
        
        try:
            if match.group(1):
                day = int(match.group(1))
                month_part = match.group(2).lower()
                year_part = match.group(3)
                month = self.months.get(month_part)
                if month is None:
                    return False
            else:
                day = int(match.group(4))
                month = int(match.group(6))
                year_part = match.group(7)
            
            if year_part:
                year = int(year_part)
                if year < 100:
                    year += 2000
            else:
                year = result.reference_datetime.year
            
            parsed_date = self._datetime_for_date(
                result.reference_datetime,
                year,
                month,
                day,
            )
            
            # Extract time if present in same expression
            time_result = self._extract_time(expr)
            if time_result:
                hours, minutes = time_result
                parsed_date = parsed_date.replace(hour=hours, minute=minutes)
                self._set_datetime(result, parsed_date)
            else:
                self._set_date(result, parsed_date)
                self._apply_time_of_day_interval(result, expr)
            result.confidence = 0.95
            result.notes.append("Parsed as explicit date")
            
            return True
            
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse explicit date: {e}")
            return False

    def _parse_day_of_month(
        self,
        expr: str,
        result: NormalizedTemporal,
    ) -> bool:
        """Resolve an unqualified day number to its next valid occurrence."""
        match = self.patterns.DAY_OF_MONTH_PATTERN.search(expr)
        if not match:
            return False

        day = int(match.group(1))
        time_result = self._extract_time(expr)
        reference = result.reference_datetime

        for month_offset in range(13):
            month_index = reference.month - 1 + month_offset
            year = reference.year + month_index // 12
            month = month_index % 12 + 1
            try:
                candidate = self._datetime_for_date(reference, year, month, day)
            except ValueError:
                continue

            if time_result:
                candidate = candidate.replace(hour=time_result[0], minute=time_result[1])
                if candidate <= reference:
                    continue
                self._set_datetime(result, candidate)
            else:
                if candidate.date() < reference.date():
                    continue
                self._set_date(result, candidate)
                self._apply_time_of_day_interval(result, expr)

            result.confidence = 0.88
            result.notes.append("Month inferred as the next occurrence of the stated day")
            return True

        return False
    
    def _parse_weekday(
        self,
        expr: str,
        result: NormalizedTemporal
    ) -> bool:
        """
        Parse weekday expressions (e.g., "segunda", "próxima sexta", "esta terça").
        
        Returns True if successfully parsed.
        """
        match = self.patterns.WEEKDAY_PATTERN.search(expr)
        if not match:
            return False
        
        try:
            qualifier = match.group(1)
            weekday_name = match.group(2).lower()
            
            # Get weekday number (0=Monday, 6=Sunday)
            target_weekday = self.weekdays.get(weekday_name)
            if target_weekday is None:
                return False
            
            # Current reference info
            ref_date = result.reference_datetime.date()
            current_weekday = result.reference_datetime.weekday()
            
            # Calculate target date
            if qualifier and qualifier.lower() in WEEKDAY_QUALIFIERS:
                qualifier_type = WEEKDAY_QUALIFIERS[qualifier.lower()]
                
                if qualifier_type == 'next':
                    # Next occurrence of weekday
                    days_ahead = target_weekday - current_weekday
                    if days_ahead <= 0:  # Target day already happened
                        days_ahead += 7
                    target_date = ref_date + timedelta(days=days_ahead)
                    result.notes.append("Parsed as next occurrence of weekday")
                
                elif qualifier_type == 'this':
                    # This week's occurrence
                    days_ahead = target_weekday - current_weekday
                    if days_ahead < 0:  # Already happened
                        days_ahead += 7
                    elif days_ahead == 0:  # Today
                        days_ahead = 0
                    target_date = ref_date + timedelta(days=days_ahead)
                    result.notes.append("Parsed as this week's occurrence")
            else:
                # No qualifier - default to next occurrence (or today if same weekday)
                days_ahead = target_weekday - current_weekday
                if days_ahead < 0:
                    days_ahead += 7
                elif days_ahead == 0:
                    # Same weekday - could be today or next week (ambiguity)
                    # Default: if time is specified and later, assume today; else assume next week
                    time_match = self._extract_time(expr)
                    if time_match:
                        hours, minutes = time_match
                        current_time = result.reference_datetime.time()
                        if time_obj(hours, minutes) > current_time:
                            days_ahead = 0  # Later today
                        else:
                            days_ahead = 7  # Next week
                            result.notes.append("Ambiguous weekday with past time; assuming next week")
                    else:
                        days_ahead = 7  # Assume next week by default
                        result.notes.append("Ambiguous weekday without time; assuming next week")
                
                target_date = ref_date + timedelta(days=days_ahead)
            
            parsed_datetime = datetime.combine(
                target_date,
                time_obj(0, 0),
                tzinfo=result.reference_datetime.tzinfo,
            )
            
            # Extract time if present
            time_result = self._extract_time(expr)
            if time_result:
                hours, minutes = time_result
                parsed_datetime = parsed_datetime.replace(hour=hours, minute=minutes)
                self._set_datetime(result, parsed_datetime)
            else:
                self._set_date(result, parsed_datetime)
            result.confidence = 0.9
            
            return True
            
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse weekday: {e}")
            return False
    
    def _parse_relative_expression(
        self,
        expr: str,
        result: NormalizedTemporal
    ) -> bool:
        """
        Parse relative expressions (e.g., "amanhã", "hoje", "para a semana").
        
        Returns True if successfully parsed.
        """
        match = self.patterns.RELATIVE_PATTERN.search(expr)
        if not match:
            return False
        
        try:
            rel_expr = match.group(1).lower()
            value = self.relative_expr.get(rel_expr)
            
            if value is None:
                return False
            
            ref_datetime = result.reference_datetime
            target_datetime: Optional[datetime] = None

            if isinstance(value, int):
                target_datetime = (ref_datetime + timedelta(days=value)).replace(
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
                time_result = self._extract_time(expr)
                if time_result:
                    target_datetime = target_datetime.replace(
                        hour=time_result[0],
                        minute=time_result[1],
                    )
                    self._set_datetime(result, target_datetime)
                else:
                    self._set_date(
                        result,
                        target_datetime,
                        temporal_type=TemporalType.RELATIVE,
                    )

            elif value == 'this_week':
                monday = ref_datetime.date() - timedelta(days=ref_datetime.weekday())
                start = datetime.combine(monday, time_obj(0, 0), tzinfo=ref_datetime.tzinfo)
                end = start + timedelta(days=7) - timedelta(seconds=1)
                self._set_interval(result, start, end, "week")

            elif value == 'next_week':
                monday = ref_datetime.date() - timedelta(days=ref_datetime.weekday())
                start = datetime.combine(
                    monday + timedelta(days=7),
                    time_obj(0, 0),
                    tzinfo=ref_datetime.tzinfo,
                )
                end = start + timedelta(days=7) - timedelta(seconds=1)
                self._set_interval(result, start, end, "week")

            elif value == 'this_month':
                first_day = ref_datetime.replace(
                    day=1,
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
                next_month = first_day + timedelta(days=32)
                end = next_month.replace(day=1) - timedelta(seconds=1)
                self._set_interval(result, first_day, end, "month")

            elif value == 'next_month':
                next_month_date = ref_datetime.replace(day=28) + timedelta(days=4)
                first_day = next_month_date.replace(
                    day=1,
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
                next_next_month = first_day + timedelta(days=32)
                end = next_next_month.replace(day=1) - timedelta(seconds=1)
                self._set_interval(result, first_day, end, "month")

            elif value == 'end_of_week':
                sunday = ref_datetime.date() + timedelta(days=6 - ref_datetime.weekday())
                target_datetime = datetime.combine(
                    sunday,
                    time_obj(0, 0),
                    tzinfo=ref_datetime.tzinfo,
                )
                self._set_date(result, target_datetime)

            elif value == 'soon':
                start = ref_datetime.replace(microsecond=0)
                self._set_interval(result, start, start + timedelta(days=3), "vague")
                result.confidence = 0.7
                result.notes.append("'Soon' is vague; assuming next 3 days")
            
            result.confidence = 0.85 if result.temporal_type != TemporalType.INTERVAL else 0.75
            result.notes.append(f"Parsed relative expression: '{rel_expr}'")
            
            return True
            
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse relative expression: {e}")
            return False

    def _parse_relative_offset(
        self,
        expr: str,
        result: NormalizedTemporal,
    ) -> bool:
        """Parse numeric offsets such as 'daqui a 2 dias às 15h'."""
        match = self.patterns.RELATIVE_OFFSET_PATTERN.search(expr)
        if not match:
            return False

        amount = int(match.group(1))
        unit = match.group(2).lower()
        if unit.startswith("semana"):
            delta = timedelta(weeks=amount)
        elif unit.startswith("hora"):
            delta = timedelta(hours=amount)
        else:
            delta = timedelta(days=amount)

        target = result.reference_datetime + delta
        time_result = self._extract_time(expr)
        if time_result:
            target = target.replace(hour=time_result[0], minute=time_result[1])
            self._set_datetime(result, target)
        elif unit.startswith("hora"):
            self._set_datetime(result, target, precision="hour")
        else:
            self._set_date(
                result,
                target,
                temporal_type=TemporalType.RELATIVE,
            )

        result.confidence = 0.9
        result.notes.append(f"Parsed relative offset: {amount} {unit}")
        return True
    
    def _parse_complex_expression(
        self,
        expr: str,
        result: NormalizedTemporal
    ) -> bool:
        """
        Parse complex expressions combining multiple parts (e.g., "sexta à tarde às 15h").
        Strategy: combine date/weekday with time of day and/or specific time.
        
        Returns True if successfully parsed.
        """
        
        # Example: "sexta à tarde" or "sexta às 15h"
        # Try weekday + time of day
        weekday_match = self.patterns.WEEKDAY_PATTERN.search(expr)
        time_of_day_match = self.patterns.TIME_OF_DAY_PATTERN.search(expr)
        
        if weekday_match and time_of_day_match:
            try:
                if not self._parse_weekday(expr, result):
                    return False

                # An explicit clock time is more precise than a broad period.
                if self._extract_time(expr):
                    return True
                
                time_of_day_expr = time_of_day_match.group(0).lower()
                time_range = self._get_time_of_day_range(time_of_day_expr)
                
                if time_range and result.normalized_date:
                    start_h, start_m, end_h, end_m = time_range
                    base_date = datetime.fromisoformat(result.normalized_date)
                    start = base_date.replace(
                        hour=start_h,
                        minute=start_m,
                        tzinfo=result.reference_datetime.tzinfo,
                    )
                    end = base_date.replace(
                        hour=end_h,
                        minute=end_m,
                        tzinfo=result.reference_datetime.tzinfo,
                    )
                    self._set_interval(result, start, end, "time_of_day")
                    result.confidence = 0.75
                    result.notes.append(
                        f"Combined weekday with time-of-day interval: '{time_of_day_expr}'"
                    )
                    
                    return True
            
            except Exception as e:
                logger.debug(f"Failed to parse complex expression: {e}")
                return False
        
        # Try relative expression + specific time
        relative_match = self.patterns.RELATIVE_PATTERN.search(expr)
        if relative_match:
            try:
                if self._parse_relative_expression(expr, result):
                    if (
                        result.temporal_type != TemporalType.INTERVAL
                        and not self._extract_time(expr)
                    ):
                        self._apply_time_of_day_interval(result, expr)
                    return True
            except Exception as e:
                logger.debug(f"Failed parsing relative in complex: {e}")
        
        return False
    
    def _parse_time_of_day_standalone(
        self,
        expr: str,
        result: NormalizedTemporal
    ) -> bool:
        """
        Parse standalone time-of-day expressions (e.g., "manhã", "à tarde", "depois de almoço").
        Uses reference date and applies time of day range.
        
        Returns True if successfully parsed.
        """
        # Check if it's ONLY a time-of-day expression (not combined with weekday/date)
        match = self.patterns.TIME_OF_DAY_PATTERN.search(expr)
        if not match:
            return False
        
        # Make sure there's no date/weekday component
        if (self.patterns.EXPLICIT_DATE_PATTERN.search(expr) or
            self.patterns.WEEKDAY_PATTERN.search(expr) or
            self.patterns.RELATIVE_PATTERN.search(expr)):
            return False  # This is combinatorial, handle elsewhere
        
        try:
            time_of_day_expr = match.group(0).lower()
            time_range = self._get_time_of_day_range(time_of_day_expr)
            
            if time_range:
                start_h, start_m, end_h, end_m = time_range
                start = result.reference_datetime.replace(
                    hour=start_h,
                    minute=start_m,
                    second=0,
                    microsecond=0,
                )
                end = result.reference_datetime.replace(
                    hour=end_h,
                    minute=end_m,
                    second=0,
                    microsecond=0,
                )
                self._set_interval(result, start, end, "time_of_day")
                result.confidence = 0.75
                result.notes.append(f"Time-of-day interval: '{time_of_day_expr}'")
                
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Failed to parse time of day: {e}")
            return False
    
    def _parse_time_only(
        self,
        expr: str,
        result: NormalizedTemporal
    ) -> bool:
        """
        Parse time-only expressions (e.g., "15h", "às 14:30").
        Uses reference date and only updates time component.
        
        Returns True if successfully parsed.
        """
        time_result = self._extract_time(expr)
        if not time_result:
            return False
        
        try:
            hours, minutes = time_result
            target_datetime = result.reference_datetime.replace(
                hour=hours,
                minute=minutes,
                second=0
            )
            
            self._set_datetime(
                result,
                target_datetime,
                precision="time_only",
                temporal_type=TemporalType.TIME,
            )
            result.confidence = 0.8
            result.notes.append("Parsed as time-only expression")
            
            return True
            
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse time-only: {e}")
            return False
    
    def _extract_time(self, expr: str) -> Optional[Tuple[int, int]]:
        """
        Extract hours and minutes from expression.
        
        Returns:
            Tuple of (hours, minutes) or None if no time found
        """
        match = self.patterns.TIME_PATTERN.search(expr)
        if not match:
            return None
        
        try:
            hours = int(match.group(1))
            minute_group = match.group(2) or match.group(3)
            minutes = int(minute_group) if minute_group else 0
            
            # Validate ranges
            if not (0 <= hours <= 23):
                return None
            if not (0 <= minutes <= 59):
                return None
            
            return (hours, minutes)
            
        except (ValueError, TypeError):
            return None
    
    def _get_time_of_day_range(self, time_of_day_expr: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Get time range (start_h, start_m, end_h, end_m) for time of day expression.
        
        Returns:
            Tuple of (start_h, start_m, end_h, end_m) or None
        """
        expr_lower = time_of_day_expr.lower()
        
        for key, value in self.time_of_day.items():
            if key in expr_lower:
                return value
        
        return None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def normalize_temporal(
    temporal_expression: str,
    reference_datetime: datetime = None,
    context: Optional[Dict] = None,
) -> NormalizedTemporal:
    """
    Convenience function to normalize a temporal expression.
    
    Args:
        temporal_expression: Expression to normalize
        reference_datetime: Reference point (default: now)
        context: Optional context dictionary
    
    Returns:
        NormalizedTemporal object
    """
    normalizer = TemporalNormalizer()
    return normalizer.normalize(temporal_expression, reference_datetime, context)


def batch_normalize_temporals(
    temporal_expressions: List[str],
    reference_datetime: datetime = None,
) -> List[NormalizedTemporal]:
    """
    Normalize multiple temporal expressions.
    
    Args:
        temporal_expressions: List of expressions to normalize
        reference_datetime: Reference point (default: now)
    
    Returns:
        List of NormalizedTemporal objects
    """
    normalizer = TemporalNormalizer()
    return [
        normalizer.normalize(expr, reference_datetime)
        for expr in temporal_expressions
    ]
