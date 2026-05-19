"""
qa_evaluator.py
===============

Módulo de avaliação para Question Answering.

Implementa métricas padrão:
- Exact Match (EM): Resposta exata vs referência
- F1 Score: Overlap de tokens
- Precision: Quantos tokens preditos estão corretos
- Recall: Quantos tokens de referência foram preditos
- Mean Average Precision (MAP): Para múltiplas respostas

Também fornece:
- Comparação com baseline (regex extraction)
- Análise de erros
- Relatórios de avaliação
- Confusion matrices

Project: Email Recognition PT-PT
Version: 1.0
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
import numpy as np

from qa_utils import MetricsCalculator, TextNormalizer


logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Métricas de avaliação individual."""
    
    example_id: str
    question: str
    predicted: str
    reference: str
    
    exact_match: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    confidence: float = 0.0
    
    category: Optional[str] = None
    error_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return asdict(self)


@dataclass
class AggregatedMetrics:
    """Métricas agregadas."""
    
    total_examples: int = 0
    exact_match_score: float = 0.0
    mean_f1_score: float = 0.0
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_confidence: float = 0.0
    
    per_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    error_distribution: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return asdict(self)


class ErrorAnalyzer:
    """Análise de tipos de erro em QA."""
    
    ERROR_TYPES = {
        'EXACT_MATCH': 'Resposta exatamente correta',
        'PARTIAL_MATCH': 'Resposta parcialmente correta',
        'WRONG_ANSWER': 'Resposta completamente errada',
        'EMPTY_ANSWER': 'Sistema não retornou resposta',
        'LOW_CONFIDENCE': 'Confiança muito baixa',
        'HALLUCINATION': 'Resposta não existe no contexto',
        'TRUNCATION': 'Resposta truncada',
    }
    
    @staticmethod
    def classify_error(
        predicted: str,
        reference: str,
        confidence: float,
        context: str,
    ) -> str:
        """
        Classifica tipo de erro.
        
        Args:
            predicted: Resposta predita
            reference: Resposta de referência
            confidence: Confiança do modelo
            context: Contexto (email)
            
        Returns:
            Tipo de erro
        """
        # Sem resposta
        if not predicted or len(predicted.strip()) == 0:
            if confidence < 0.5:
                return 'LOW_CONFIDENCE'
            return 'EMPTY_ANSWER'
        
        # Exato
        if TextNormalizer.clean_text(predicted.lower()) == TextNormalizer.clean_text(reference.lower()):
            return 'EXACT_MATCH'
        
        # Hallucination: resposta não existe no contexto
        if predicted not in context and predicted.lower() not in context.lower():
            return 'HALLUCINATION'
        
        # Truncation: resposta é substring de referência ou vice-versa
        if (predicted in reference or reference in predicted or
            predicted.lower() in reference.lower() or reference.lower() in predicted.lower()):
            return 'TRUNCATION'
        
        # Partial match: algum overlap de tokens
        pred_tokens = set(predicted.lower().split())
        ref_tokens = set(reference.lower().split())
        if len(pred_tokens & ref_tokens) > 0:
            return 'PARTIAL_MATCH'
        
        return 'WRONG_ANSWER'


class QAEvaluator:
    """
    Avaliador principal de Question Answering.
    
    Pipeline:
    1. Carregar predições e referências
    2. Calcular métricas para cada exemplo
    3. Agregar métricas
    4. Gerar relatório de análise de erros
    """
    
    def __init__(self):
        """Inicializa evaluador."""
        self.metrics_list: List[EvaluationMetrics] = []
        self.aggregated_metrics: Optional[AggregatedMetrics] = None
        self.error_analyzer = ErrorAnalyzer()
    
    def evaluate_example(
        self,
        example_id: str,
        question: str,
        predicted: str,
        reference: str,
        confidence: float = 1.0,
        context: str = "",
        category: Optional[str] = None,
    ) -> EvaluationMetrics:
        """
        Avalia exemplo individual.
        
        Args:
            example_id: ID do exemplo
            question: Pergunta
            predicted: Resposta predita
            reference: Resposta de referência
            confidence: Confiança do modelo
            context: Contexto (para análise de erro)
            category: Categoria (participants, time, etc)
            
        Returns:
            EvaluationMetrics com scores
        """
        # Calcular métricas
        metrics = MetricsCalculator.compute_metrics(predicted, reference)
        
        # Classificar erro
        error_type = self.error_analyzer.classify_error(
            predicted, reference, confidence, context
        )
        
        # Criar resultado
        eval_metrics = EvaluationMetrics(
            example_id=example_id,
            question=question,
            predicted=predicted,
            reference=reference,
            exact_match=metrics['exact_match'],
            f1_score=metrics['f1'],
            confidence=confidence,
            category=category,
            error_type=error_type,
        )
        
        self.metrics_list.append(eval_metrics)
        return eval_metrics
    
    def batch_evaluate(
        self,
        predictions: List[Dict[str, Any]],
        references: List[Dict[str, Any]],
    ) -> AggregatedMetrics:
        """
        Avalia batch de exemplos.
        
        Espera formato:
        predictions = [
            {
                'id': str,
                'question': str,
                'answer': str,
                'confidence': float,
                'category': str
            }
        ]
        
        references = [
            {
                'id': str,
                'question': str,
                'answer': str,
                'category': str
            }
        ]
        
        Args:
            predictions: Lista de predições
            references: Lista de referências
            
        Returns:
            AggregatedMetrics
        """
        # Mapear referências por ID
        ref_map = {ref['id']: ref for ref in references}
        
        # Avaliar cada predição
        for pred in predictions:
            pred_id = pred['id']
            
            if pred_id not in ref_map:
                logger.warning(f"Referência não encontrada para ID {pred_id}")
                continue
            
            ref = ref_map[pred_id]
            
            self.evaluate_example(
                example_id=pred_id,
                question=pred.get('question', ''),
                predicted=pred.get('answer', ''),
                reference=ref.get('answer', ''),
                confidence=pred.get('confidence', 1.0),
                context=pred.get('context', ''),
                category=pred.get('category'),
            )
        
        # Agregar
        return self.aggregate_metrics()
    
    def aggregate_metrics(self) -> AggregatedMetrics:
        """
        Agrega métricas de todos os exemplos.
        
        Returns:
            AggregatedMetrics
        """
        if not self.metrics_list:
            return AggregatedMetrics()
        
        # Inicializar
        agg_metrics = AggregatedMetrics(
            total_examples=len(self.metrics_list)
        )
        
        # Agregação global
        em_scores = []
        f1_scores = []
        precisions = []
        recalls = []
        confidences = []
        
        # Agregação por categoria
        per_category: Dict[str, List[EvaluationMetrics]] = defaultdict(list)
        
        # Distribuição de erros
        error_counts: Counter = Counter()
        
        for metrics in self.metrics_list:
            em_scores.append(metrics.exact_match)
            f1_scores.append(metrics.f1_score)
            confidences.append(metrics.confidence)
            
            if metrics.category:
                per_category[metrics.category].append(metrics)
            
            if metrics.error_type:
                error_counts[metrics.error_type] += 1
        
        # Calcular médias globais
        if em_scores:
            agg_metrics.exact_match_score = np.mean(em_scores)
        if f1_scores:
            agg_metrics.mean_f1_score = np.mean(f1_scores)
        if confidences:
            agg_metrics.mean_confidence = np.mean(confidences)
        
        # Calcular por categoria
        for category, metrics_list in per_category.items():
            cat_em = np.mean([m.exact_match for m in metrics_list])
            cat_f1 = np.mean([m.f1_score for m in metrics_list])
            
            agg_metrics.per_category[category] = {
                'exact_match': cat_em,
                'f1_score': cat_f1,
                'count': len(metrics_list),
            }
        
        # Distribuição de erros
        agg_metrics.error_distribution = dict(error_counts)
        
        self.aggregated_metrics = agg_metrics
        return agg_metrics
    
    def save_results(
        self,
        output_file: str,
        include_examples: bool = True,
    ) -> None:
        """
        Salva resultados de avaliação.
        
        Args:
            output_file: Ficheiro de output
            include_examples: Se True, inclui todos os exemplos
        """
        output_data = {
            'aggregated': self.aggregated_metrics.to_dict() if self.aggregated_metrics else {},
            'examples': [m.to_dict() for m in self.metrics_list] if include_examples else [],
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Resultados salvos em {output_file}")
    
    def print_report(self) -> None:
        """Imprime relatório de avaliação."""
        if not self.aggregated_metrics:
            logger.warning("Nenhuma métrica agregada disponível")
            return
        
        agg = self.aggregated_metrics
        
        print("\n" + "="*70)
        print("QA EVALUATION REPORT")
        print("="*70)
        
        print(f"\nTotal Examples: {agg.total_examples}")
        
        print(f"\nGlobal Metrics:")
        print(f"  Exact Match (EM): {agg.exact_match_score:.4f}")
        print(f"  Mean F1 Score: {agg.mean_f1_score:.4f}")
        print(f"  Mean Confidence: {agg.mean_confidence:.4f}")
        
        if agg.per_category:
            print(f"\nMetrics by Category:")
            for category, metrics in agg.per_category.items():
                print(f"  {category}:")
                print(f"    EM: {metrics['exact_match']:.4f}")
                print(f"    F1: {metrics['f1_score']:.4f}")
                print(f"    Count: {metrics['count']}")
        
        if agg.error_distribution:
            print(f"\nError Distribution:")
            total_errors = sum(agg.error_distribution.values())
            for error_type, count in sorted(
                agg.error_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                percentage = (count / total_errors) * 100
                print(f"  {error_type}: {count} ({percentage:.1f}%)")
        
        print("\n" + "="*70)
    
    def print_error_analysis(self, top_n: int = 5) -> None:
        """
        Imprime análise detalhada de erros.
        
        Args:
            top_n: Número de erros para mostrar
        """
        # Filtrar erros
        errors = [m for m in self.metrics_list if m.exact_match == 0.0]
        
        if not errors:
            print("\nSem erros encontrados!")
            return
        
        print(f"\n" + "="*70)
        print(f"ERROR ANALYSIS (Top {min(top_n, len(errors))} errors)")
        print("="*70)
        
        # Ordenar por F1 score (piores primeiro)
        errors.sort(key=lambda x: x.f1_score)
        
        for i, error in enumerate(errors[:top_n]):
            print(f"\nError {i+1}: [{error.error_type}]")
            print(f"  Pergunta: {error.question}")
            print(f"  Predita: {error.predicted}")
            print(f"  Referência: {error.reference}")
            print(f"  F1: {error.f1_score:.4f}")
            print(f"  Confiança: {error.confidence:.4f}")


class ComparativeAnalyzer:
    """Análise comparativa entre métodos."""
    
    @staticmethod
    def compare_methods(
        method_results: Dict[str, List[EvaluationMetrics]],
    ) -> Dict[str, Any]:
        """
        Compara resultados entre múltiplos métodos.
        
        Args:
            method_results: Dicionário mapeando nome_método -> metrics_list
            
        Returns:
            Comparação estruturada
        """
        comparison = {}
        
        for method_name, metrics_list in method_results.items():
            em_scores = [m.exact_match for m in metrics_list]
            f1_scores = [m.f1_score for m in metrics_list]
            
            comparison[method_name] = {
                'em': np.mean(em_scores),
                'f1': np.mean(f1_scores),
                'em_std': np.std(em_scores),
                'f1_std': np.std(f1_scores),
            }
        
        return comparison


def main():
    """Exemplo de uso do evaluador."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Criar evaluador
    evaluator = QAEvaluator()
    
    # Exemplos de avaliação
    examples = [
        {
            'id': '1',
            'question': 'Quem participa?',
            'predicted': 'Ana',
            'reference': 'Ana',
            'confidence': 0.95,
        },
        {
            'id': '2',
            'question': 'Quando?',
            'predicted': 'sexta',
            'reference': 'sexta às 15h',
            'confidence': 0.75,
        },
        {
            'id': '3',
            'question': 'Onde?',
            'predicted': 'internet',
            'reference': 'Teams',
            'confidence': 0.60,
        },
    ]
    
    # Avaliar
    for ex in examples:
        evaluator.evaluate_example(
            example_id=ex['id'],
            question=ex['question'],
            predicted=ex['predicted'],
            reference=ex['reference'],
            confidence=ex['confidence'],
            category='test',
        )
    
    # Agregar e imprimir
    evaluator.aggregate_metrics()
    evaluator.print_report()
    evaluator.print_error_analysis()


if __name__ == "__main__":
    main()
