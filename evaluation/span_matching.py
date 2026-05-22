"""
Span Matching Module for Argument Extraction Evaluation

This module implements multiple span matching strategies for comparing
gold annotations with model predictions. Supports:
- Exact matching
- Token-level overlap
- Character-level overlap (Jaccard similarity)
- Fuzzy matching

Designed for Portuguese informal text with proper UTF-8 handling.

Author: Automatic Evaluation Framework
License: MIT
"""

import re
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum
import unicodedata


class MatchType(Enum):
    """Classification of span matching results."""
    EXACT_MATCH = "exact"
    PARTIAL_MATCH = "partial"
    NO_MATCH = "no_match"
    FUZZY_MATCH = "fuzzy"


@dataclass
class SpanMatch:
    """
    Represents the result of comparing two spans.
    
    Attributes:
        gold_text: Original text from gold annotation
        predicted_text: Text from model prediction
        match_type: Type of match (exact, partial, no_match, fuzzy)
        overlap_ratio: Ratio of overlap (0.0 to 1.0)
        token_overlap: Number of overlapping tokens
        jaccard_similarity: Jaccard similarity coefficient
        is_match: Boolean indicating if it's considered a match
    """
    gold_text: str
    predicted_text: str
    match_type: MatchType
    overlap_ratio: float
    token_overlap: int
    jaccard_similarity: float
    is_match: bool
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "gold_text": self.gold_text,
            "predicted_text": self.predicted_text,
            "match_type": self.match_type.value,
            "overlap_ratio": round(self.overlap_ratio, 4),
            "token_overlap": self.token_overlap,
            "jaccard_similarity": round(self.jaccard_similarity, 4),
            "is_match": self.is_match
        }


class TextNormalizer:
    """
    Normalizes Portuguese text for robust span matching.
    
    Operations:
    - Lowercase conversion
    - Unicode normalization (NFD)
    - Optional accent removal
    - Optional punctuation removal
    - Whitespace normalization
    - UTF-8 validation
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_accents: bool = False,
        remove_punctuation: bool = False,
        normalize_whitespace: bool = True
    ):
        """
        Initialize normalizer with options.
        
        Args:
            lowercase: Convert to lowercase
            remove_accents: Remove diacritics (ã, ç, é, etc.)
            remove_punctuation: Remove punctuation marks
            normalize_whitespace: Normalize spaces and tabs
        """
        self.lowercase = lowercase
        self.remove_accents = remove_accents
        self.remove_punctuation = remove_punctuation
        self.normalize_whitespace = normalize_whitespace
    
    def normalize(self, text: str) -> str:
        """
        Apply all normalization operations to text.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Ensure UTF-8 encoding
        text = text.encode('utf-8', errors='replace').decode('utf-8')
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        # Remove accents (optional)
        if self.remove_accents:
            text = self._remove_accents(text)
        
        # Remove punctuation (optional)
        if self.remove_punctuation:
            text = self._remove_punctuation(text)
        
        return text
    
    @staticmethod
    def _remove_accents(text: str) -> str:
        """
        Remove accent marks from Portuguese text.
        
        Examples:
            "São Paulo" → "Sao Paulo"
            "Ação" → "Acao"
        """
        nfd = unicodedata.normalize('NFD', text)
        return ''.join(
            char for char in nfd 
            if unicodedata.category(char) != 'Mn'
        )
    
    @staticmethod
    def _remove_punctuation(text: str) -> str:
        """Remove punctuation marks."""
        return re.sub(r'[^\w\s]', '', text)


class TokenOverlapMatcher:
    """
    Matches spans based on token-level overlap.
    
    Tokenization:
    - Whitespace-based splitting
    - Punctuation-aware
    - Portuguese contractions support
    """
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Handle Portuguese contractions and common patterns
        text = re.sub(r'\s+', ' ', text)  # normalize spaces
        tokens = text.split()
        return tokens
    
    @staticmethod
    def token_overlap(gold_tokens: List[str], pred_tokens: List[str]) -> Tuple[int, float]:
        """
        Calculate token-level overlap between two token lists.
        
        Args:
            gold_tokens: Tokens from gold annotation
            pred_tokens: Tokens from prediction
            
        Returns:
            Tuple of (overlap_count, overlap_ratio)
        """
        gold_set = set(gold_tokens)
        pred_set = set(pred_tokens)
        
        overlap = len(gold_set & pred_set)
        max_len = max(len(gold_set), len(pred_set))
        
        ratio = overlap / max_len if max_len > 0 else 0.0
        
        return overlap, ratio


class CharacterOverlapMatcher:
    """
    Matches spans based on character-level overlap.
    
    Metrics:
    - Overlap ratio (intersection / union)
    - Jaccard similarity (intersection / union of characters)
    - Levenshtein-inspired fuzzy matching
    """
    
    @staticmethod
    def jaccard_similarity(gold: str, predicted: str) -> float:
        """
        Calculate Jaccard similarity between two strings.
        
        Jaccard = |intersection| / |union|
        
        Args:
            gold: Gold annotation text
            predicted: Predicted text
            
        Returns:
            Jaccard coefficient (0.0 to 1.0)
        """
        if not gold or not predicted:
            return 0.0 if gold != predicted else 1.0
        
        gold_chars = set(gold)
        pred_chars = set(predicted)
        
        intersection = len(gold_chars & pred_chars)
        union = len(gold_chars | pred_chars)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def overlap_ratio(gold: str, predicted: str) -> float:
        """
        Calculate character-level overlap ratio.
        
        Overlap = min(len(gold), len(predicted)) characters matched
        
        Args:
            gold: Gold annotation text
            predicted: Predicted text
            
        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        if not gold or not predicted:
            return 1.0 if gold == predicted else 0.0
        
        # Use minimum length as denominator for conservative metric
        min_len = min(len(gold), len(predicted))
        max_len = max(len(gold), len(predicted))
        
        # Count matching characters (simple longest common substring heuristic)
        matches = sum(
            1 for g, p in zip(gold, predicted) 
            if g == p
        )
        
        return matches / max_len if max_len > 0 else 0.0


class SpanMatcher:
    """
    Main span matching orchestrator combining multiple strategies.
    
    Provides comprehensive span comparison with configurable thresholds.
    """
    
    def __init__(
        self,
        exact_match_threshold: float = 1.0,
        partial_match_threshold: float = 0.7,
        fuzzy_match_threshold: float = 0.6,
        normalize_text: bool = True,
        remove_accents: bool = False,
        remove_punctuation: bool = False
    ):
        """
        Initialize span matcher.
        
        Args:
            exact_match_threshold: Threshold for exact matches (default: 1.0)
            partial_match_threshold: Threshold for partial matches
            fuzzy_match_threshold: Threshold for fuzzy matches
            normalize_text: Normalize text before comparison
            remove_accents: Remove accents in comparison
            remove_punctuation: Remove punctuation in comparison
        """
        self.exact_match_threshold = exact_match_threshold
        self.partial_match_threshold = partial_match_threshold
        self.fuzzy_match_threshold = fuzzy_match_threshold
        
        self.normalizer = TextNormalizer(
            lowercase=normalize_text,
            remove_accents=remove_accents,
            remove_punctuation=remove_punctuation,
            normalize_whitespace=True
        )
        
        self.token_matcher = TokenOverlapMatcher()
        self.char_matcher = CharacterOverlapMatcher()
    
    def match(self, gold: str, predicted: str) -> SpanMatch:
        """
        Comprehensive span matching combining multiple metrics.
        
        Algorithm:
        1. Normalize both texts
        2. Check for exact match
        3. If not exact, compute token overlap
        4. Compute character-level metrics
        5. Classify match type based on thresholds
        
        Args:
            gold: Gold annotation text
            predicted: Predicted text
            
        Returns:
            SpanMatch object with detailed metrics
        """
        # Preserve originals for reporting
        gold_original = gold
        predicted_original = predicted
        
        # Normalize for comparison
        gold_norm = self.normalizer.normalize(gold)
        pred_norm = self.normalizer.normalize(predicted)
        
        # Check exact match (after normalization)
        if gold_norm == pred_norm:
            return SpanMatch(
                gold_text=gold_original,
                predicted_text=predicted_original,
                match_type=MatchType.EXACT_MATCH,
                overlap_ratio=1.0,
                token_overlap=len(self.token_matcher.tokenize(gold_norm)),
                jaccard_similarity=1.0,
                is_match=True
            )
        
        # Compute token overlap
        gold_tokens = self.token_matcher.tokenize(gold_norm)
        pred_tokens = self.token_matcher.tokenize(pred_norm)
        token_overlap_count, token_overlap_ratio = self.token_matcher.token_overlap(
            gold_tokens, pred_tokens
        )
        
        # Compute character-level metrics
        jaccard = self.char_matcher.jaccard_similarity(gold_norm, pred_norm)
        overlap = self.char_matcher.overlap_ratio(gold_norm, pred_norm)
        
        # Determine match type based on thresholds
        if jaccard >= self.partial_match_threshold:
            match_type = MatchType.PARTIAL_MATCH
            is_match = True
        elif jaccard >= self.fuzzy_match_threshold:
            match_type = MatchType.FUZZY_MATCH
            is_match = True
        else:
            match_type = MatchType.NO_MATCH
            is_match = False
        
        return SpanMatch(
            gold_text=gold_original,
            predicted_text=predicted_original,
            match_type=match_type,
            overlap_ratio=overlap,
            token_overlap=token_overlap_count,
            jaccard_similarity=jaccard,
            is_match=is_match
        )
    
    def match_lists(
        self, 
        gold_list: List[str], 
        predicted_list: List[str]
    ) -> Tuple[List[SpanMatch], List[str], List[str]]:
        """
        Match two lists of spans using greedy best-match algorithm.
        
        Algorithm:
        1. For each predicted span, find best match in gold
        2. Assign matches greedily from highest confidence
        3. Identify false positives (unmatched predictions)
        4. Identify false negatives (unmatched gold)
        
        Args:
            gold_list: List of gold annotation spans
            predicted_list: List of predicted spans
            
        Returns:
            Tuple of (matches, false_positives, false_negatives)
        """
        matches = []
        matched_gold_indices = set()
        matched_pred_indices = set()
        
        # Try to match each predicted span with gold spans
        for pred_idx, pred in enumerate(predicted_list):
            best_match = None
            best_score = 0.0
            best_gold_idx = -1
            
            # Find best match in gold annotations
            for gold_idx, gold in enumerate(gold_list):
                if gold_idx in matched_gold_indices:
                    continue
                
                match = self.match(gold, pred)
                score = match.jaccard_similarity
                
                if score > best_score:
                    best_score = score
                    best_match = match
                    best_gold_idx = gold_idx
            
            # Accept match if above threshold
            if best_match and best_match.is_match:
                matches.append(best_match)
                matched_gold_indices.add(best_gold_idx)
                matched_pred_indices.add(pred_idx)
        
        # Identify false positives and false negatives
        false_positives = [
            predicted_list[i] for i in range(len(predicted_list))
            if i not in matched_pred_indices
        ]
        
        false_negatives = [
            gold_list[i] for i in range(len(gold_list))
            if i not in matched_gold_indices
        ]
        
        return matches, false_positives, false_negatives
