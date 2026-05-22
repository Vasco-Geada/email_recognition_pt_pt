"""
Argument Extraction Evaluator

Main orchestrator for evaluating argument extraction models.
Coordinates span matching, metrics calculation, and error analysis.

Features:
- Multi-argument-type evaluation
- Per-email and aggregated metrics
- Error analysis and categorization
- Support for multiple match types
- Portuguese language handling

Author: Automatic Evaluation Framework
License: MIT
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime

from span_matching import SpanMatcher, MatchType
from metrics import (
    MetricsCalculator, 
    MatchAwareMetricsCalculator,
    ClassMetrics, 
    AggregatedMetrics,
    AggregationMethod,
    EvaluationReport
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Argument type definitions
ARGUMENT_TYPES = ["participants", "time", "location", "topic"]


@dataclass
class EmailEvaluationResult:
    """
    Evaluation result for a single email.
    
    Attributes:
        email_id: Unique identifier
        text: Email text (for reference)
        intent: Email intent category
        per_argument_metrics: Metrics per argument type
        error_analysis: Detailed error information
        match_details: Information about each match
    """
    email_id: int
    text: str
    intent: str
    per_argument_metrics: Dict[str, Dict] = field(default_factory=dict)
    error_analysis: Dict[str, Any] = field(default_factory=dict)
    match_details: Dict[str, List[Dict]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "email_id": self.email_id,
            "text": self.text,
            "intent": self.intent,
            "per_argument_metrics": self.per_argument_metrics,
            "error_analysis": self.error_analysis,
            "match_details": self.match_details
        }


@dataclass
class ErrorInstance:
    """
    Single error instance for analysis.
    
    Attributes:
        email_id: Email identifier
        argument_type: Type of argument (participants, time, etc.)
        error_type: Type of error (false_positive, false_negative, partial_match)
        gold_value: Gold annotation value
        predicted_value: Predicted value (if applicable)
        context: Additional context information
    """
    email_id: int
    argument_type: str
    error_type: str  # "false_positive", "false_negative", "partial_match"
    gold_value: Optional[str] = None
    predicted_value: Optional[str] = None
    context: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "email_id": self.email_id,
            "argument_type": self.argument_type,
            "error_type": self.error_type,
            "gold_value": self.gold_value,
            "predicted_value": self.predicted_value,
            "context": self.context
        }


class ArgumentExtractionEvaluator:
    """
    Main evaluator for argument extraction models.
    
    Workflow:
    1. Load gold annotations and predictions
    2. Initialize span matcher with normalization options
    3. For each email and argument type:
       a. Match spans between gold and predicted
       b. Calculate confusion metrics
       c. Identify and categorize errors
    4. Aggregate metrics across all emails
    5. Generate reports and analysis
    """
    
    def __init__(
        self,
        exact_match_threshold: float = 1.0,
        partial_match_threshold: float = 0.7,
        fuzzy_match_threshold: float = 0.6,
        normalize_text: bool = True,
        remove_accents: bool = False,
        remove_punctuation: bool = False,
        verbose: bool = True
    ):
        """
        Initialize the evaluator.
        
        Args:
            exact_match_threshold: Threshold for exact matches
            partial_match_threshold: Threshold for partial matches
            fuzzy_match_threshold: Threshold for fuzzy matches
            normalize_text: Normalize text before comparison
            remove_accents: Remove accents in comparison
            remove_punctuation: Remove punctuation in comparison
            verbose: Print progress information
        """
        self.span_matcher = SpanMatcher(
            exact_match_threshold=exact_match_threshold,
            partial_match_threshold=partial_match_threshold,
            fuzzy_match_threshold=fuzzy_match_threshold,
            normalize_text=normalize_text,
            remove_accents=remove_accents,
            remove_punctuation=remove_punctuation
        )
        
        self.verbose = verbose
        self.email_results: List[EmailEvaluationResult] = []
        self.errors: List[ErrorInstance] = []
        self.metrics_calculator = MetricsCalculator()
    
    def evaluate_single_email(
        self,
        email_id: int,
        text: str,
        intent: str,
        gold_arguments: Dict[str, List[str]],
        predicted_arguments: Dict[str, List[str]]
    ) -> EmailEvaluationResult:
        """
        Evaluate a single email's argument extraction.
        
        Args:
            email_id: Unique identifier
            text: Email text
            intent: Email intent category
            gold_arguments: Gold annotations {arg_type: [spans]}
            predicted_arguments: Predictions {arg_type: [spans]}
            
        Returns:
            EmailEvaluationResult with detailed metrics and error analysis
        """
        result = EmailEvaluationResult(
            email_id=email_id,
            text=text,
            intent=intent
        )
        
        # Evaluate each argument type
        for arg_type in ARGUMENT_TYPES:
            gold = gold_arguments.get(arg_type, [])
            pred = predicted_arguments.get(arg_type, [])
            
            # Match spans
            matches, false_positives, false_negatives = self.span_matcher.match_lists(
                gold, pred
            )
            
            # Calculate metrics for this argument type
            tp = len([m for m in matches if m.is_match])
            fp = len(false_positives)
            fn = len(false_negatives)
            
            class_metric = self.metrics_calculator.calculate_class_metrics(
                arg_type, tp, fp, fn
            )
            
            result.per_argument_metrics[arg_type] = {
                "precision": class_metric.precision,
                "recall": class_metric.recall,
                "f1": class_metric.f1,
                "support": class_metric.support,
                "tp": tp,
                "fp": fp,
                "fn": fn
            }
            
            # Store match details
            result.match_details[arg_type] = [m.to_dict() for m in matches]
            
            # Record errors
            self._record_errors(
                email_id, arg_type, matches, false_positives, false_negatives
            )
        
        self.email_results.append(result)
        return result
    
    def _record_errors(
        self,
        email_id: int,
        argument_type: str,
        matches: List,
        false_positives: List[str],
        false_negatives: List[str]
    ) -> None:
        """
        Record errors for analysis.
        
        Args:
            email_id: Email identifier
            argument_type: Type of argument
            matches: List of SpanMatch objects
            false_positives: List of incorrectly predicted spans
            false_negatives: List of missed spans
        """
        # Record false negatives (gold spans not predicted)
        for fn_text in false_negatives:
            error = ErrorInstance(
                email_id=email_id,
                argument_type=argument_type,
                error_type="false_negative",
                gold_value=fn_text,
                predicted_value=None
            )
            self.errors.append(error)
        
        # Record false positives (predicted spans not in gold)
        for fp_text in false_positives:
            error = ErrorInstance(
                email_id=email_id,
                argument_type=argument_type,
                error_type="false_positive",
                gold_value=None,
                predicted_value=fp_text
            )
            self.errors.append(error)
        
        # Record partial matches
        for match in matches:
            if match.match_type == MatchType.PARTIAL_MATCH:
                error = ErrorInstance(
                    email_id=email_id,
                    argument_type=argument_type,
                    error_type="partial_match",
                    gold_value=match.gold_text,
                    predicted_value=match.predicted_text,
                    context={
                        "jaccard_similarity": match.jaccard_similarity,
                        "overlap_ratio": match.overlap_ratio,
                        "token_overlap": match.token_overlap
                    }
                )
                self.errors.append(error)
    
    def evaluate_batch(
        self,
        emails_data: List[Dict]
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of emails.
        
        Expected format for each email:
        {
            "id": int,
            "text": str,
            "intent": str,
            "arguments": {"participants": [], "time": [], "location": [], "topic": []},
            "predicted": {"participants": [], "time": [], "location": [], "topic": []}
        }
        
        Args:
            emails_data: List of email evaluation data
            
        Returns:
            Dictionary with aggregated results
        """
        if self.verbose:
            logger.info(f"Evaluating {len(emails_data)} emails...")
        
        for email_data in emails_data:
            try:
                self.evaluate_single_email(
                    email_id=email_data["id"],
                    text=email_data["text"],
                    intent=email_data["intent"],
                    gold_arguments=email_data["arguments"],
                    predicted_arguments=email_data.get("predicted", {})
                )
            except Exception as e:
                logger.error(f"Error evaluating email {email_data.get('id')}: {str(e)}")
                continue
        
        return self.get_aggregated_results()
    
    def get_aggregated_results(self) -> Dict[str, Any]:
        """
        Compute aggregated metrics across all evaluated emails.
        
        Returns:
            Dictionary with comprehensive evaluation results
        """
        if not self.email_results:
            logger.warning("No emails have been evaluated yet")
            return {}
        
        # Aggregate metrics per argument type
        per_argument_metrics: Dict[str, ClassMetrics] = {}
        
        for arg_type in ARGUMENT_TYPES:
            total_tp = 0
            total_fp = 0
            total_fn = 0
            
            for result in self.email_results:
                metrics = result.per_argument_metrics.get(arg_type, {})
                total_tp += metrics.get("tp", 0)
                total_fp += metrics.get("fp", 0)
                total_fn += metrics.get("fn", 0)
            
            class_metric = self.metrics_calculator.calculate_class_metrics(
                arg_type, total_tp, total_fp, total_fn
            )
            per_argument_metrics[arg_type] = class_metric
        
        # Calculate aggregated metrics
        aggregated = self.metrics_calculator.aggregate_metrics(
            per_argument_metrics,
            method=AggregationMethod.MICRO
        )
        
        # Create report
        report = EvaluationReport(
            model_name="default_model",
            per_class_metrics=per_argument_metrics,
            aggregated_metrics=aggregated
        )
        
        # Error summary
        error_summary = self._summarize_errors()
        
        return {
            "evaluation_report": report.to_dict(),
            "email_results": [r.to_dict() for r in self.email_results],
            "error_analysis": error_summary,
            "summary_statistics": {
                "total_emails": len(self.email_results),
                "total_errors": len(self.errors),
                "false_positives": sum(1 for e in self.errors if e.error_type == "false_positive"),
                "false_negatives": sum(1 for e in self.errors if e.error_type == "false_negative"),
                "partial_matches": sum(1 for e in self.errors if e.error_type == "partial_match")
            }
        }
    
    def _summarize_errors(self) -> Dict[str, Any]:
        """
        Summarize errors by type and argument class.
        
        Returns:
            Dictionary with error analysis
        """
        error_by_type: Dict[str, List[Dict]] = {
            "false_positive": [],
            "false_negative": [],
            "partial_match": []
        }
        
        error_by_argument: Dict[str, Dict] = {arg: {"count": 0, "examples": []} for arg in ARGUMENT_TYPES}
        
        for error in self.errors:
            # Group by error type
            error_by_type[error.error_type].append(error.to_dict())
            
            # Group by argument type
            error_by_argument[error.argument_type]["count"] += 1
            if len(error_by_argument[error.argument_type]["examples"]) < 3:
                error_by_argument[error.argument_type]["examples"].append(error.to_dict())
        
        return {
            "by_type": error_by_type,
            "by_argument": error_by_argument,
            "top_error_arguments": sorted(
                error_by_argument.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )[:5]
        }
    
    def save_results(self, output_path: str) -> None:
        """
        Save evaluation results to JSON file.
        
        Args:
            output_path: Path to save results
        """
        results = self.get_aggregated_results()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            logger.info(f"Results saved to {output_path}")
    
    def save_errors(self, output_path: str) -> None:
        """
        Save detailed error analysis to JSON file.
        
        Args:
            output_path: Path to save errors
        """
        errors_data = {
            "timestamp": datetime.now().isoformat(),
            "total_errors": len(self.errors),
            "errors": [e.to_dict() for e in self.errors],
            "summary": self._summarize_errors()
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(errors_data, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            logger.info(f"Errors saved to {output_path}")
