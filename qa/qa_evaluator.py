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
import argparse
import csv
import numpy as np

from qa_utils import MetricsCalculator, TextNormalizer
from qa_questions import QAQuestions, QuestionCategory


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
            precision=metrics['precision'],
            recall=metrics['recall'],
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
            precisions.append(metrics.precision)
            recalls.append(metrics.recall)
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
        if precisions:
            agg_metrics.mean_precision = np.mean(precisions)
        if recalls:
            agg_metrics.mean_recall = np.mean(recalls)
        if confidences:
            agg_metrics.mean_confidence = np.mean(confidences)
        
        # Calcular por categoria
        for category, metrics_list in per_category.items():
            cat_em = np.mean([m.exact_match for m in metrics_list])
            cat_f1 = np.mean([m.f1_score for m in metrics_list])
            cat_precision = np.mean([m.precision for m in metrics_list])
            cat_recall = np.mean([m.recall for m in metrics_list])
            cat_confidence = np.mean([m.confidence for m in metrics_list])
            
            agg_metrics.per_category[category] = {
                'exact_match': cat_em,
                'f1_score': cat_f1,
                'precision': cat_precision,
                'recall': cat_recall,
                'mean_confidence': cat_confidence,
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
        print(f"  Mean Precision: {agg.mean_precision:.4f}")
        print(f"  Mean Recall: {agg.mean_recall:.4f}")
        print(f"  Mean Confidence: {agg.mean_confidence:.4f}")
        
        if agg.per_category:
            print(f"\nMetrics by Category:")
            for category, metrics in agg.per_category.items():
                print(f"  {category}:")
                print(f"    EM: {metrics['exact_match']:.4f}")
                print(f"    F1: {metrics['f1_score']:.4f}")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall: {metrics['recall']:.4f}")
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


class ProjectQAEvaluation:
    """Avalia outputs do QAPipeline contra gold annotations do projeto."""

    CATEGORIES = ["participants", "time", "time_normalized", "location", "topic"]

    @classmethod
    def evaluate_files(
        cls,
        gold_file: str,
        predictions_file: str,
        output_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> QAEvaluator:
        gold = cls._load_json(gold_file)
        predictions = cls._load_json(predictions_file)

        evaluator = QAEvaluator()
        gold_by_id, gold_by_shifted_id, gold_by_text = cls._build_gold_indexes(gold)

        missing_gold = []
        for prediction in predictions:
            gold_item = cls._find_gold_item(
                prediction,
                gold_by_id=gold_by_id,
                gold_by_shifted_id=gold_by_shifted_id,
                gold_by_text=gold_by_text,
            )
            if gold_item is None:
                missing_gold.append(prediction.get("email_id"))
                continue

            cls._evaluate_prediction(evaluator, prediction, gold_item)

        evaluator.aggregate_metrics()

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            evaluator.save_results(str(output_path / "qa_metrics.json"))
            cls.write_metrics_csv(evaluator, output_path / "qa_metrics.csv")
            cls.write_summary_csv(evaluator, output_path / "qa_summary.csv")

            metadata = {
                "gold_file": gold_file,
                "predictions_file": predictions_file,
                "evaluated_examples": len(evaluator.metrics_list),
                "missing_gold_ids": missing_gold,
            }
            with open(output_path / "qa_evaluation_metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        if verbose:
            evaluator.print_report()
            if missing_gold:
                logger.warning("Sem gold annotation para email_ids: %s", missing_gold)

        return evaluator

    @staticmethod
    def _load_json(path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Esperava lista JSON em {path}")
        return data

    @classmethod
    def _build_gold_indexes(
        cls,
        gold: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Dict]]:
        by_id = {}
        by_shifted_id = {}
        by_text = {}

        for item in gold:
            item_id = item.get("id")
            if item_id is not None:
                by_id[str(item_id)] = item
                if isinstance(item_id, int) and item_id > 0:
                    by_shifted_id[str(item_id - 1)] = item

            text = cls._normalize_for_lookup(item.get("text", ""))
            if text:
                by_text[text] = item

        return by_id, by_shifted_id, by_text

    @classmethod
    def _find_gold_item(
        cls,
        prediction: Dict[str, Any],
        gold_by_id: Dict[str, Dict],
        gold_by_shifted_id: Dict[str, Dict],
        gold_by_text: Dict[str, Dict],
    ) -> Optional[Dict]:
        pred_id = prediction.get("email_id")
        if pred_id is not None:
            pred_key = str(pred_id)
            if pred_key in gold_by_id:
                return gold_by_id[pred_key]
            if pred_key in gold_by_shifted_id:
                return gold_by_shifted_id[pred_key]

        text = cls._normalize_for_lookup(prediction.get("email_text", ""))
        return gold_by_text.get(text)

    @classmethod
    def _evaluate_prediction(
        cls,
        evaluator: QAEvaluator,
        prediction: Dict[str, Any],
        gold_item: Dict[str, Any],
    ) -> None:
        qa_results = prediction.get("qa_results", {})
        gold_arguments = gold_item.get("arguments", {})
        example_id = str(prediction.get("email_id", gold_item.get("id", "")))
        context = prediction.get("email_text", gold_item.get("text", ""))

        for category in cls.CATEGORIES:
            if category == "time_normalized":
                predicted_result = cls._normalized_time_result(qa_results.get("time", {}) or {})
            else:
                predicted_result = qa_results.get(category, {}) or {}
            predicted_answer = predicted_result.get("answer") or ""
            references = cls._as_reference_list(gold_arguments.get(category, []))
            question = predicted_result.get("question") or cls._question_for_category(category)
            confidence = float(predicted_result.get("confidence") or 0.0)

            best_reference, best_scores = cls._best_reference_match(predicted_answer, references)
            evaluator.evaluate_example(
                example_id=f"{example_id}:{category}",
                question=question,
                predicted=predicted_answer,
                reference=best_reference,
                confidence=confidence,
                context=context,
                category=category,
            )

            last_metric = evaluator.metrics_list[-1]
            last_metric.exact_match = best_scores["exact_match"]
            last_metric.precision = best_scores["precision"]
            last_metric.recall = best_scores["recall"]
            last_metric.f1_score = best_scores["f1"]

    @staticmethod
    def _as_reference_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if str(value).strip():
            return [str(value)]
        return []

    @staticmethod
    def _normalized_time_result(time_result: Dict[str, Any]) -> Dict[str, Any]:
        normalized = time_result.get("normalized") or {}
        answer = (
            normalized.get("normalized_datetime")
            or normalized.get("interval_start")
            or normalized.get("normalized_date")
            or ""
        )
        return {
            "answer": answer,
            "confidence": time_result.get("confidence", 0.0),
            "question": "Qual e a data/hora normalizada da reuniao?",
            "valid": bool(answer),
        }

    @staticmethod
    def _best_reference_match(predicted: str, references: List[str]) -> Tuple[str, Dict[str, float]]:
        if not references:
            metrics = MetricsCalculator.compute_metrics(predicted or "", "")
            return "", metrics

        scored = [
            (reference, MetricsCalculator.compute_metrics(predicted or "", reference))
            for reference in references
        ]
        return max(
            scored,
            key=lambda item: (item[1]["exact_match"], item[1]["f1"], item[1]["recall"]),
        )

    @staticmethod
    def _question_for_category(category: str) -> str:
        mapping = {
            "participants": QuestionCategory.PARTICIPANTS,
            "time": QuestionCategory.TIME,
            "location": QuestionCategory.LOCATION,
            "topic": QuestionCategory.TOPIC,
        }
        if category == "time_normalized":
            return "Qual e a data/hora normalizada da reuniao?"
        return QAQuestions.get_primary_question(mapping[category])

    @staticmethod
    def _normalize_for_lookup(text: str) -> str:
        return TextNormalizer.clean_text(str(text), lowercase=True)

    @staticmethod
    def write_metrics_csv(evaluator: QAEvaluator, output_file: Path) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "example_id",
            "category",
            "exact_match",
            "precision",
            "recall",
            "f1_score",
            "confidence",
            "predicted",
            "reference",
            "error_type",
        ]
        with output_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for metric in evaluator.metrics_list:
                writer.writerow({column: getattr(metric, column) for column in columns})

    @staticmethod
    def write_summary_csv(evaluator: QAEvaluator, output_file: Path) -> None:
        if evaluator.aggregated_metrics is None:
            evaluator.aggregate_metrics()

        agg = evaluator.aggregated_metrics
        output_file.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "scope",
            "category",
            "count",
            "exact_match",
            "precision",
            "recall",
            "f1_score",
            "mean_confidence",
        ]

        with output_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerow(
                {
                    "scope": "global",
                    "category": "all",
                    "count": agg.total_examples,
                    "exact_match": agg.exact_match_score,
                    "precision": agg.mean_precision,
                    "recall": agg.mean_recall,
                    "f1_score": agg.mean_f1_score,
                    "mean_confidence": agg.mean_confidence,
                }
            )
            for category, metrics in agg.per_category.items():
                category_confidences = [
                    item.confidence for item in evaluator.metrics_list
                    if item.category == category
                ]
                writer.writerow(
                    {
                        "scope": "category",
                        "category": category,
                        "count": metrics["count"],
                        "exact_match": metrics["exact_match"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1_score": metrics["f1_score"],
                        "mean_confidence": float(np.mean(category_confidences))
                        if category_confidences else 0.0,
                    }
                )


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


def cli_main():
    """CLI para avaliar resultados QA contra gold annotations."""
    parser = argparse.ArgumentParser(
        description="Avalia qa_results.json contra gold annotations do projeto."
    )
    parser.add_argument(
        "--gold",
        required=True,
        help="Ficheiro gold JSON com arguments participants/time/location/topic.",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Ficheiro qa_results.json gerado pelo QAPipeline.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Diretorio onde guardar qa_metrics.json/csv.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Nao imprimir relatorio no terminal.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    ProjectQAEvaluation.evaluate_files(
        gold_file=args.gold,
        predictions_file=args.predictions,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    cli_main()
