"""
Argument Extraction Evaluation Framework

Comprehensive evaluation framework for comparing argument extraction models
in Portuguese emails related to meeting scheduling.

Key Features:
- Multi-argument-type evaluation (participants, time, location, topic)
- Multiple span matching strategies (exact, partial, fuzzy)
- Comprehensive metrics (Precision, Recall, F1)
- Per-argument-type analysis
- Error analysis and categorization
- Model comparison
- Multiple output formats (JSON, CSV, Markdown, LaTeX)
- Visualizations (bar charts, confusion matrices, heatmaps)

Main Components:
1. span_matching: Span matching strategies
2. metrics: Metrics calculation
3. evaluate_arguments: Main evaluator
4. report_generator: Report generation
5. visualization: Visualizations
6. model_comparison: Model comparison
7. utils: Utility functions

Usage Example:
    from evaluation import ArgumentExtractionEvaluator, ModelComparator
    
    # Single model evaluation
    evaluator = ArgumentExtractionEvaluator()
    results = evaluator.evaluate_batch(emails_data)
    evaluator.save_results("results.json")
    
    # Multiple models comparison
    comparator = ModelComparator()
    comparator.register_model("model1", evaluator1)
    comparator.register_model("model2", evaluator2)
    results = comparator.evaluate_all_models(emails_data)
    comparator.print_comparison_summary()

Author: Automatic Evaluation Framework
License: MIT
Version: 1.0.0
"""

from .span_matching import (
    SpanMatcher,
    SpanMatch,
    MatchType,
    TextNormalizer,
    TokenOverlapMatcher,
    CharacterOverlapMatcher
)

from .metrics import (
    MetricsCalculator,
    MatchAwareMetricsCalculator,
    ClassMetrics,
    AggregatedMetrics,
    AggregationMethod,
    ConfusionMetrics,
    EvaluationReport
)

from .evaluate_arguments import (
    ArgumentExtractionEvaluator,
    EmailEvaluationResult,
    ErrorInstance
)

from .report_generator import ReportGenerator

from .visualization import Visualizer

from .model_comparison import ModelComparator

from .utils import (
    DataLoader,
    DataValidator,
    DataPreprocessor,
    EvaluationUtils
)

__all__ = [
    # Span matching
    "SpanMatcher",
    "SpanMatch",
    "MatchType",
    "TextNormalizer",
    "TokenOverlapMatcher",
    "CharacterOverlapMatcher",
    
    # Metrics
    "MetricsCalculator",
    "MatchAwareMetricsCalculator",
    "ClassMetrics",
    "AggregatedMetrics",
    "AggregationMethod",
    "ConfusionMetrics",
    "EvaluationReport",
    
    # Evaluation
    "ArgumentExtractionEvaluator",
    "EmailEvaluationResult",
    "ErrorInstance",
    
    # Reports and visualization
    "ReportGenerator",
    "Visualizer",
    "ModelComparator",
    
    # Utilities
    "DataLoader",
    "DataValidator",
    "DataPreprocessor",
    "EvaluationUtils"
]

__version__ = "1.0.0"
__author__ = "Automatic Evaluation Framework"
__license__ = "MIT"
