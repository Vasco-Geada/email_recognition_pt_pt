"""
Question Answering Module for Email Recognition PT-PT

A complete QA pipeline for extracting structured information from
informal Portuguese academic emails about meetings.

Modules:
--------
- qa_questions: Definition of QA questions and categories
- qa_utils: Utilities (text processing, metrics, caching)
- qa_dataset_generator: Convert gold annotations to QA format
- qa_inference: HuggingFace transformers inference engine
- qa_evaluator: Evaluation metrics and analysis
- qa_pipeline: Integrated QA pipeline

Quick Start:
-----------
>>> from qa.qa_pipeline import QAPipeline
>>>
>>> pipeline = QAPipeline(model_name='bertimbau-pt')
>>> result = pipeline.process_email(
...     email_text="Boas Ana, reunimos sexta às 15h no Teams?"
... )
>>> print(result.get_answers_only())
{
    'participants': 'Ana',
    'time': 'sexta às 15h',
    'location': 'Teams',
    'topic': None
}


Project: Email Recognition PT-PT
Version: 1.0
"""

__version__ = "1.0.0"
__author__ = "NLP Engineer"

# Import main components
from qa_questions import (
    QAQuestions,
    QuestionCategory,
    Question,
)

from qa_utils import (
    QAResult,
    TextNormalizer,
    AnswerPostProcessor,
    ConfidenceScaler,
    MetricsCalculator,
)

from qa_dataset_generator import (
    QADataset,
    QAExample,
    QADatasetGenerator,
)

from qa_inference import (
    QAInferenceEngine,
    QAModelLoader,
    QAResultsCache,
    MultilingualQAFallback,
)

from qa_evaluator import (
    QAEvaluator,
    EvaluationMetrics,
    AggregatedMetrics,
    ErrorAnalyzer,
)

from qa_pipeline import (
    QAPipeline,
    EmailQAResult,
    QuickQA,
)

__all__ = [
    # Questions
    'QAQuestions',
    'QuestionCategory',
    'Question',
    
    # Utils
    'QAResult',
    'TextNormalizer',
    'AnswerPostProcessor',
    'ConfidenceScaler',
    'MetricsCalculator',
    
    # Dataset
    'QADataset',
    'QAExample',
    'QADatasetGenerator',
    
    # Inference
    'QAInferenceEngine',
    'QAModelLoader',
    'QAResultsCache',
    'MultilingualQAFallback',
    
    # Evaluation
    'QAEvaluator',
    'EvaluationMetrics',
    'AggregatedMetrics',
    'ErrorAnalyzer',
    
    # Pipeline
    'QAPipeline',
    'EmailQAResult',
    'QuickQA',
]
