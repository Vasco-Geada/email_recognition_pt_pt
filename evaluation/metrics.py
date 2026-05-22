"""
Metrics Module for Argument Extraction Evaluation

Implements comprehensive evaluation metrics for argument extraction tasks:
- Precision, Recall, F1-Score (token and span level)
- Micro and Macro averages
- Per-argument-type metrics
- Confusion metrics (TP, FP, FN, TN)

Compliant with academic evaluation standards (CoNLL, SemEval).

Author: Automatic Evaluation Framework
License: MIT
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import math


class AggregationMethod(Enum):
    """Aggregation methods for multi-class metrics."""
    MICRO = "micro"      # Pool all predictions, then compute
    MACRO = "macro"      # Compute per-class, then average
    WEIGHTED = "weighted"  # Weighted by support


@dataclass
class ConfusionMetrics:
    """
    Confusion matrix elements for single class/argument type.
    
    Attributes:
        tp: True Positives (correctly predicted)
        fp: False Positives (incorrectly predicted)
        fn: False Negatives (missed predictions)
        tn: True Negatives (correctly rejected) - optional for span tasks
    """
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ClassMetrics:
    """
    Evaluation metrics for a single argument type or class.
    
    Attributes:
        name: Class/argument type name (e.g., "participants", "time")
        precision: TP / (TP + FP)
        recall: TP / (TP + FN)
        f1: Harmonic mean of precision and recall
        support: Total number of gold instances
        confusion: Confusion matrix elements
    """
    name: str
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    support: int = 0
    confusion: ConfusionMetrics = field(default_factory=ConfusionMetrics)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "support": self.support,
            "confusion": self.confusion.to_dict()
        }


@dataclass
class AggregatedMetrics:
    """
    Aggregated metrics across all classes/argument types.
    
    Attributes:
        micro_precision: Micro-averaged precision
        micro_recall: Micro-averaged recall
        micro_f1: Micro-averaged F1
        macro_precision: Macro-averaged precision
        macro_recall: Macro-averaged recall
        macro_f1: Macro-averaged F1
        weighted_f1: Weighted F1 (by support)
        accuracy: Overall accuracy
        total_instances: Total gold instances
    """
    micro_precision: float = 0.0
    micro_recall: float = 0.0
    micro_f1: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    accuracy: float = 0.0
    total_instances: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "micro": {
                "precision": round(self.micro_precision, 4),
                "recall": round(self.micro_recall, 4),
                "f1": round(self.micro_f1, 4)
            },
            "macro": {
                "precision": round(self.macro_precision, 4),
                "recall": round(self.macro_recall, 4),
                "f1": round(self.macro_f1, 4)
            },
            "weighted_f1": round(self.weighted_f1, 4),
            "accuracy": round(self.accuracy, 4),
            "total_instances": self.total_instances
        }


class MetricsCalculator:
    """
    Calculates evaluation metrics for argument extraction.
    
    Supports:
    - Span-level evaluation (exact and partial matches)
    - Token-level evaluation
    - BIO tag evaluation
    - Per-argument-type metrics
    - Aggregated metrics (micro, macro, weighted)
    """
    
    @staticmethod
    def precision(tp: int, fp: int) -> float:
        """
        Calculate precision: TP / (TP + FP)
        
        Args:
            tp: True Positives
            fp: False Positives
            
        Returns:
            Precision score (0.0 to 1.0), or 0.0 if no predictions
        """
        denominator = tp + fp
        return tp / denominator if denominator > 0 else 0.0
    
    @staticmethod
    def recall(tp: int, fn: int) -> float:
        """
        Calculate recall: TP / (TP + FN)
        
        Args:
            tp: True Positives
            fn: False Negatives
            
        Returns:
            Recall score (0.0 to 1.0), or 0.0 if no gold instances
        """
        denominator = tp + fn
        return tp / denominator if denominator > 0 else 0.0
    
    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        """
        Calculate F1 score: 2 * (precision * recall) / (precision + recall)
        
        Args:
            precision: Precision score
            recall: Recall score
            
        Returns:
            F1 score (0.0 to 1.0), or 0.0 if both are 0
        """
        denominator = precision + recall
        return 2 * (precision * recall) / denominator if denominator > 0 else 0.0
    
    @staticmethod
    def accuracy(correct: int, total: int) -> float:
        """
        Calculate accuracy: correct / total
        
        Args:
            correct: Number of correct predictions
            total: Total number of instances
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        return correct / total if total > 0 else 0.0
    
    @classmethod
    def calculate_class_metrics(
        cls,
        class_name: str,
        tp: int,
        fp: int,
        fn: int,
        tn: int = 0
    ) -> ClassMetrics:
        """
        Calculate comprehensive metrics for a single class/argument type.
        
        Args:
            class_name: Name of the class (e.g., "participants")
            tp: True Positives
            fp: False Positives
            fn: False Negatives
            tn: True Negatives (optional)
            
        Returns:
            ClassMetrics object with all metrics
        """
        precision = cls.precision(tp, fp)
        recall = cls.recall(tp, fn)
        f1 = cls.f1_score(precision, recall)
        support = tp + fn  # Total gold instances
        
        return ClassMetrics(
            name=class_name,
            precision=precision,
            recall=recall,
            f1=f1,
            support=support,
            confusion=ConfusionMetrics(tp=tp, fp=fp, fn=fn, tn=tn)
        )
    
    @classmethod
    def aggregate_metrics(
        cls,
        class_metrics: Dict[str, ClassMetrics],
        method: AggregationMethod = AggregationMethod.MICRO
    ) -> AggregatedMetrics:
        """
        Aggregate metrics across multiple classes.
        
        Methods:
        - MICRO: Pool all TPs, FPs, FNs, then compute metrics
        - MACRO: Compute metrics per class, then average
        - WEIGHTED: Macro average weighted by support (number of gold instances)
        
        Args:
            class_metrics: Dictionary mapping class names to ClassMetrics
            method: Aggregation method
            
        Returns:
            AggregatedMetrics object
        """
        if not class_metrics:
            return AggregatedMetrics()
        
        agg = AggregatedMetrics()
        
        if method == AggregationMethod.MICRO:
            # Pool all confusion elements
            total_tp = sum(m.confusion.tp for m in class_metrics.values())
            total_fp = sum(m.confusion.fp for m in class_metrics.values())
            total_fn = sum(m.confusion.fn for m in class_metrics.values())
            
            agg.micro_precision = cls.precision(total_tp, total_fp)
            agg.micro_recall = cls.recall(total_tp, total_fn)
            agg.micro_f1 = cls.f1_score(agg.micro_precision, agg.micro_recall)
            agg.total_instances = total_tp + total_fn
            
        elif method == AggregationMethod.MACRO:
            # Average metrics per class
            precisions = [m.precision for m in class_metrics.values()]
            recalls = [m.recall for m in class_metrics.values()]
            f1s = [m.f1 for m in class_metrics.values()]
            
            n_classes = len(class_metrics)
            agg.macro_precision = sum(precisions) / n_classes if n_classes > 0 else 0.0
            agg.macro_recall = sum(recalls) / n_classes if n_classes > 0 else 0.0
            agg.macro_f1 = sum(f1s) / n_classes if n_classes > 0 else 0.0
            agg.total_instances = sum(m.support for m in class_metrics.values())
        
        elif method == AggregationMethod.WEIGHTED:
            # Weighted average by support
            f1s = [m.f1 for m in class_metrics.values()]
            supports = [m.support for m in class_metrics.values()]
            total_support = sum(supports)
            
            if total_support > 0:
                agg.weighted_f1 = sum(f * s for f, s in zip(f1s, supports)) / total_support
            else:
                agg.weighted_f1 = 0.0
            
            agg.total_instances = total_support
        
        return agg
    
    @staticmethod
    def calculate_accuracy_from_metrics(
        class_metrics: Dict[str, ClassMetrics]
    ) -> float:
        """
        Calculate overall accuracy from class metrics.
        
        Accuracy = total correct / total instances
        
        Args:
            class_metrics: Dictionary mapping class names to ClassMetrics
            
        Returns:
            Overall accuracy score
        """
        total_tp = sum(m.confusion.tp for m in class_metrics.values())
        total_instances = sum(m.support for m in class_metrics.values())
        
        return total_tp / total_instances if total_instances > 0 else 0.0


class MatchAwareMetricsCalculator(MetricsCalculator):
    """
    Extension of MetricsCalculator that handles multiple match types.
    
    Supports:
    - Exact match only
    - Exact + Partial matches
    - Exact + Partial + Fuzzy matches
    """
    
    @staticmethod
    def calculate_with_match_types(
        exact_count: int,
        partial_count: int,
        fuzzy_count: int,
        false_positives: int,
        false_negatives: int,
        match_type: str = "exact"
    ) -> Tuple[float, float, float]:
        """
        Calculate metrics considering different match types.
        
        Args:
            exact_count: Number of exact matches
            partial_count: Number of partial matches
            fuzzy_count: Number of fuzzy matches
            false_positives: Number of false positives
            false_negatives: Number of false negatives
            match_type: Which match types to count as correct ("exact", "partial", or "all")
            
        Returns:
            Tuple of (precision, recall, f1)
        """
        if match_type == "exact":
            tp = exact_count
        elif match_type == "partial":
            tp = exact_count + partial_count
        elif match_type == "all":
            tp = exact_count + partial_count + fuzzy_count
        else:
            tp = exact_count
        
        precision = MetricsCalculator.precision(tp, false_positives)
        recall = MetricsCalculator.recall(tp, false_negatives)
        f1 = MetricsCalculator.f1_score(precision, recall)
        
        return precision, recall, f1


@dataclass
class EvaluationReport:
    """
    Complete evaluation report for a single model.
    
    Attributes:
        model_name: Name of the model being evaluated
        per_class_metrics: Metrics for each argument type
        aggregated_metrics: Aggregated metrics
        match_type_results: Results for different match types
    """
    model_name: str
    per_class_metrics: Dict[str, ClassMetrics] = field(default_factory=dict)
    aggregated_metrics: AggregatedMetrics = field(default_factory=AggregatedMetrics)
    match_type_results: Dict[str, Dict] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "per_class_metrics": {
                name: metrics.to_dict() 
                for name, metrics in self.per_class_metrics.items()
            },
            "aggregated_metrics": self.aggregated_metrics.to_dict(),
            "match_type_results": self.match_type_results
        }
