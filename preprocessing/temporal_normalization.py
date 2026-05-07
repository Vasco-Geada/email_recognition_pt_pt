"""
Temporal Expression Normalization Module for European Portuguese

This module normalizes temporal expressions extracted from Portuguese emails
into structured datetime objects. It supports:
- Relative expressions (amanhã, hoje, para a semana)
- Weekday expressions (segunda, terça, sexta)
- Time expressions (às 15h, 10:30)
- Informal temporal markers (depois de almoço)
- Complex combined expressions

Author: NLP Engineer
Date: 2026
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, time as time_obj
from enum import Enum

logger = logging.getLogger(__name__)


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
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with ISO string dates."""
        result = {
            'original_text': self.original_text,
            'reference_datetime': self.reference_datetime.isoformat(),
            'temporal_type': self.temporal_type.value,
            'normalized_datetime': self.normalized_datetime_str,
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
    'terça': 1,             # Tuesday
    'terça-feira': 1,
    'quarta': 2,            # Wednesday
    'quarta-feira': 2,
    'quinta': 3,            # Thursday
    'quinta-feira': 3,
    'sexta': 4,             # Friday
    'sexta-feira': 4,
    'sábado': 5,            # Saturday
    'sábado': 5,
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
    'de manhã': (6, 0, 12, 0),
    'pela manhã': (6, 0, 12, 0),
    
    # Afternoon
    'tarde': (12, 0, 18, 0),           # 12h-18h
    'de tarde': (12, 0, 18, 0),
    'pela tarde': (12, 0, 18, 0),
    
    # Evening
    'noite': (18, 0, 23, 59),          # 18h-23:59h
    'de noite': (18, 0, 23, 59),
    'pela noite': (18, 0, 23, 59),
    
    # After lunch
    'depois de almoço': (12, 30, 14, 0),  # ~12:30-14:00
    'pós almoço': (12, 30, 14, 0),
    
    # After breakfast
    'depois do café': (9, 0, 10, 30),  # ~9:00-10:30
    'pós café': (9, 0, 10, 30),
}

# Month names
MONTHS = {
    'janeiro': 1, 'jan': 1,
    'fevereiro': 2, 'fev': 2,
    'março': 3, 'mar': 3,
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


# ============================================================================
# REGEX PATTERNS
# ============================================================================

class TemporalPatterns:
    """Compiled regex patterns for temporal expression matching."""
    
    # Time patterns - more specific to avoid false positives
    # Examples: "15h", "15:30", "15h30", "às 15h", "às 15:30"
    # Must have explicit time separator (h, :, or .) to avoid matching dates
    TIME_PATTERN = re.compile(
        r'(?:às?\s*)?(\d{1,2})(?:[h:\.])(?:(\d{2}))?(?:\s*(?:min|h))?(?!\s*de)',
        re.IGNORECASE
    )
    
    # Explicit date patterns - start of line or after space/punctuation
    # Examples: "16 de Abril", "16 de abril de 2026", "16/04/2026", "16-04-2026"
    EXPLICIT_DATE_PATTERN = re.compile(
        r'(?:^|\s)'  # Start or after space
        r'(\d{1,2})\s*(?:de|/|-)?\s*'  # Day
        r'([a-záéíóú]+|\d{1,2})\s*'     # Month (name or number)
        r'(?:(?:de|/)?\s*(\d{4}))?',    # Year (optional)
        re.IGNORECASE | re.MULTILINE
    )
    
    # Weekday patterns
    # Examples: "segunda", "próxima sexta", "esta terça"
    WEEKDAY_PATTERN = re.compile(
        r'(?:(próxima|proxima|próximo|proximo|esta|este)\s+)?'  # Qualifier (optional)
        r'(segunda|terça|quarta|quinta|sexta|sábado|domingo|'
        r'segunda-feira|terça-feira|quarta-feira|quinta-feira|sexta-feira)',
        re.IGNORECASE
    )
    
    # Relative expression patterns (amanhã, hoje, ontem, etc.)
    RELATIVE_PATTERN = re.compile(
        r'(amanhã|hoje|ontem|agora|'
        r'para a semana|esta semana|próximo mês|este mês|'
        r'em breve|'
        r'depois de amanhã)',
        re.IGNORECASE
    )
    
    # Time of day patterns (manhã, tarde, noite, depois de almoço)
    TIME_OF_DAY_PATTERN = re.compile(
        r'(pela\s+)?(manhã|tarde|noite|após\s+almoço|'
        r'depois\s+de\s+almoço|pós\s+almoço|'
        r'após\s+café|depois\s+do\s+café|pós\s+café|'
        r'de\s+manhã|de\s+tarde|de\s+noite)',
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
        
        # Normalize input
        expr_normalized = temporal_expression.strip().lower()
        
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
            
            # 2. Weekday patterns
            parsed = self._parse_weekday(expr_normalized, result)
            if parsed:
                return result
            
            # 3. Relative expressions
            parsed = self._parse_relative_expression(expr_normalized, result)
            if parsed:
                return result
            
            # 4. Complex patterns (combinations)
            parsed = self._parse_complex_expression(expr_normalized, result)
            if parsed:
                return result
            
            # 5. Time of day standalone patterns
            parsed = self._parse_time_of_day_standalone(expr_normalized, result)
            if parsed:
                return result
            
            # 6. Time only patterns
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
            day = int(match.group(1))
            month_part = match.group(2).lower()
            year_part = match.group(3) if match.lastindex >= 3 else None
            
            # Parse month
            if month_part.isdigit():
                month = int(month_part)
            else:
                month = self.months.get(month_part)
                if month is None:
                    return False
            
            # Parse year (use reference year if not provided)
            if year_part:
                year = int(year_part)
            else:
                year = result.reference_datetime.year
            
            # Create datetime
            parsed_date = datetime(year, month, day)
            
            # Extract time if present in same expression
            time_result = self._extract_time(expr)
            if time_result:
                hours, minutes = time_result
                parsed_date = parsed_date.replace(hour=hours, minute=minutes)
                result.temporal_type = TemporalType.DATETIME
                result.precision = "exact"
            else:
                result.temporal_type = TemporalType.DATE
                result.precision = "day"
            
            result.normalized_datetime = parsed_date
            result.normalized_date = parsed_date.strftime('%Y-%m-%d')
            result.normalized_time = parsed_date.strftime('%H:%M:%S')
            result.normalized_datetime_str = parsed_date.isoformat()
            result.confidence = 0.95
            result.notes.append("Parsed as explicit date")
            
            return True
            
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse explicit date: {e}")
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
            
            # Create datetime
            parsed_datetime = datetime.combine(target_date, time_obj(0, 0))
            
            # Extract time if present
            time_result = self._extract_time(expr)
            if time_result:
                hours, minutes = time_result
                parsed_datetime = parsed_datetime.replace(hour=hours, minute=minutes)
                result.temporal_type = TemporalType.DATETIME
                result.precision = "exact"
            else:
                result.temporal_type = TemporalType.DATE
                result.precision = "day"
            
            result.normalized_datetime = parsed_datetime
            result.normalized_date = parsed_datetime.strftime('%Y-%m-%d')
            result.normalized_time = parsed_datetime.strftime('%H:%M:%S')
            result.normalized_datetime_str = parsed_datetime.isoformat()
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
            target_datetime = ref_datetime
            
            if isinstance(value, int):
                # Offset by days
                target_datetime = ref_datetime + timedelta(days=value)
                result.temporal_type = TemporalType.RELATIVE
                result.precision = "day"
            
            elif value == 'this_week':
                # This week: Monday to Sunday
                monday = ref_datetime.date() - timedelta(days=ref_datetime.weekday())
                target_datetime = datetime.combine(monday, time_obj(0, 0))
                result.interval_start = target_datetime
                result.interval_end = target_datetime + timedelta(days=7)
                result.interval_start_str = result.interval_start.isoformat()
                result.interval_end_str = result.interval_end.isoformat()
                result.temporal_type = TemporalType.INTERVAL
                result.precision = "week"
            
            elif value == 'next_week':
                # Next week
                monday = ref_datetime.date() - timedelta(days=ref_datetime.weekday())
                monday = monday + timedelta(days=7)  # Next Monday
                target_datetime = datetime.combine(monday, time_obj(0, 0))
                result.interval_start = target_datetime
                result.interval_end = target_datetime + timedelta(days=7)
                result.interval_start_str = result.interval_start.isoformat()
                result.interval_end_str = result.interval_end.isoformat()
                result.temporal_type = TemporalType.INTERVAL
                result.precision = "week"
            
            elif value == 'this_month':
                # This month: 1st to last day
                first_day = ref_datetime.replace(day=1)
                next_month = first_day + timedelta(days=32)
                last_day = next_month.replace(day=1) - timedelta(days=1)
                result.interval_start = first_day
                result.interval_end = datetime.combine(last_day.date(), time_obj(23, 59))
                result.interval_start_str = result.interval_start.isoformat()
                result.interval_end_str = result.interval_end.isoformat()
                result.temporal_type = TemporalType.INTERVAL
                result.precision = "month"
            
            elif value == 'next_month':
                # Next month
                next_month_date = ref_datetime + timedelta(days=32)
                first_day = next_month_date.replace(day=1)
                next_next_month = first_day + timedelta(days=32)
                last_day = next_next_month.replace(day=1) - timedelta(days=1)
                result.interval_start = first_day
                result.interval_end = datetime.combine(last_day.date(), time_obj(23, 59))
                result.interval_start_str = result.interval_start.isoformat()
                result.interval_end_str = result.interval_end.isoformat()
                result.temporal_type = TemporalType.INTERVAL
                result.precision = "month"
            
            elif value == 'soon':
                # "Soon" = next 3 days (heuristic)
                result.interval_start = ref_datetime
                result.interval_end = ref_datetime + timedelta(days=3)
                result.interval_start_str = result.interval_start.isoformat()
                result.interval_end_str = result.interval_end.isoformat()
                result.temporal_type = TemporalType.INTERVAL
                result.precision = "vague"
                result.confidence = 0.7
                result.notes.append("'Soon' is vague; assuming next 3 days")
            
            # Extract time if present
            time_result = self._extract_time(expr)
            if time_result:
                hours, minutes = time_result
                if isinstance(value, int):  # Only for day offsets
                    target_datetime = target_datetime.replace(hour=hours, minute=minutes)
                    result.temporal_type = TemporalType.DATETIME
                    result.precision = "exact"
            
            if result.temporal_type != TemporalType.INTERVAL:
                result.normalized_datetime = target_datetime
                result.normalized_date = target_datetime.strftime('%Y-%m-%d')
                result.normalized_time = target_datetime.strftime('%H:%M:%S')
                result.normalized_datetime_str = target_datetime.isoformat()
            
            result.confidence = 0.85 if result.temporal_type != TemporalType.INTERVAL else 0.75
            result.notes.append(f"Parsed relative expression: '{rel_expr}'")
            
            return True
            
        except (ValueError, TypeError) as e:
            logger.debug(f"Failed to parse relative expression: {e}")
            return False
    
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
            # Has both weekday and time of day
            try:
                # First parse weekday (without time)
                weekday_expr = weekday_match.group(0)
                if not self._parse_weekday(expr, result):
                    return False
                
                # Then apply time of day constraints
                time_of_day_expr = time_of_day_match.group(0).lower()
                time_range = self._get_time_of_day_range(time_of_day_expr)
                
                if time_range:
                    start_h, start_m, end_h, end_m = time_range
                    # Apply midpoint of time range (heuristic)
                    mid_h = (start_h + end_h) // 2
                    mid_m = (start_m + end_m) // 2
                    
                    result.normalized_datetime = result.normalized_datetime.replace(
                        hour=mid_h,
                        minute=mid_m
                    )
                    result.temporal_type = TemporalType.DATETIME
                    result.precision = "approximate"
                    result.confidence = 0.75
                    result.notes.append(f"Combined weekday with time of day: '{time_of_day_expr}'")
                    result.normalized_datetime_str = result.normalized_datetime.isoformat()
                    
                    return True
            
            except Exception as e:
                logger.debug(f"Failed to parse complex expression: {e}")
                return False
        
        # Try relative expression + specific time
        relative_match = self.patterns.RELATIVE_PATTERN.search(expr)
        if relative_match:
            try:
                if self._parse_relative_expression(expr, result):
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
                # Apply midpoint of time range (heuristic)
                mid_h = (start_h + end_h) // 2
                mid_m = (start_m + end_m) // 2
                
                target_datetime = result.reference_datetime.replace(
                    hour=mid_h,
                    minute=mid_m,
                    second=0
                )
                
                result.normalized_datetime = target_datetime
                result.normalized_time = target_datetime.strftime('%H:%M:%S')
                result.normalized_date = target_datetime.strftime('%Y-%m-%d')
                result.normalized_datetime_str = target_datetime.isoformat()
                result.temporal_type = TemporalType.TIME
                result.precision = "approximate"
                result.confidence = 0.75
                result.notes.append(f"Time of day: '{time_of_day_expr}' → ~{mid_h:02d}:{mid_m:02d}")
                
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
            
            result.normalized_datetime = target_datetime
            result.normalized_time = target_datetime.strftime('%H:%M:%S')
            result.normalized_datetime_str = target_datetime.isoformat()
            result.temporal_type = TemporalType.TIME
            result.precision = "time_only"
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
            minutes = int(match.group(2)) if match.group(2) else 0
            
            # Validate ranges
            if not (0 <= hours <= 23):
                hours = hours % 24  # Handle 24h format edge cases
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
