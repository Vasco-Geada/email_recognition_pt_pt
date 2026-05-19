"""
__init__.py
===========

Pacote de Gold Annotations para Email Recognition PT-PT.

Exporta classes principais para uso fácil:
- GoldAnnotationsGenerator
- AnnotationValidator
- AnnotationEvaluator
- HeuristicAnnotationExtractor

Author: Generated for Email Recognition PT-PT Project
"""

from .heuristic_extractors import (
    HeuristicAnnotationExtractor,
    TriggerExtractor,
    ParticipantExtractor,
    TemporalExtractor,
    LocationExtractor,
    TopicExtractor,
    ExtractionResult,
)

from .validators import (
    AnnotationValidator,
    JSONValidator,
    ConsistencyValidator,
    ValidationError,
    ValidationResult,
)

from .gold_annotations_generator import (
    GoldAnnotationsGenerator,
)

from .evaluate_annotations import (
    AnnotationEvaluator,
    EvaluationResult,
    ClassificationMetrics,
)

__version__ = "1.0.0"
__author__ = "Email Recognition PT-PT Project"
__all__ = [
    # Extractores
    'HeuristicAnnotationExtractor',
    'TriggerExtractor',
    'ParticipantExtractor',
    'TemporalExtractor',
    'LocationExtractor',
    'TopicExtractor',
    'ExtractionResult',
    
    # Validadores
    'AnnotationValidator',
    'JSONValidator',
    'ConsistencyValidator',
    'ValidationError',
    'ValidationResult',
    
    # Gerador
    'GoldAnnotationsGenerator',
    
    # Avaliador
    'AnnotationEvaluator',
    'EvaluationResult',
    'ClassificationMetrics',
]
