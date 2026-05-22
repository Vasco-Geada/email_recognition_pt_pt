"""
Model Comparison Module

Orchestrates comparison of multiple argument extraction models.

Features:
- Batch evaluation of multiple models
- Statistical comparison
- Ranking and scoring
- Differential analysis

Author: Automatic Evaluation Framework
License: MIT
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import json
from datetime import datetime

from evaluate_arguments import ArgumentExtractionEvaluator


logger = logging.getLogger(__name__)


class ModelComparator:
    """
    Compares multiple argument extraction models.
    
    Workflow:
    1. Load gold annotations and predictions from multiple models
    2. Evaluate each model independently
    3. Rank models by metrics
    4. Perform statistical comparisons
    5. Generate comparative reports
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize comparator.
        
        Args:
            verbose: Print progress information
        """
        self.verbose = verbose
        self.models_evaluators: Dict[str, ArgumentExtractionEvaluator] = {}
        self.models_results: Dict[str, Dict[str, Any]] = {}
    
    def register_model(
        self,
        model_name: str,
        evaluator: Optional[ArgumentExtractionEvaluator] = None,
        **evaluator_kwargs
    ) -> None:
        """
        Register a model for comparison.
        
        Args:
            model_name: Name of the model
            evaluator: ArgumentExtractionEvaluator instance
            **evaluator_kwargs: Arguments for creating evaluator if not provided
        """
        if evaluator is None:
            evaluator = ArgumentExtractionEvaluator(
                verbose=self.verbose,
                **evaluator_kwargs
            )
        
        self.models_evaluators[model_name] = evaluator
        
        if self.verbose:
            logger.info(f"Registered model: {model_name}")
    
    def evaluate_all_models(
        self,
        emails_data: List[Dict]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all registered models.
        
        Args:
            emails_data: List of email evaluation data
            
        Returns:
            Dictionary mapping model names to results
        """
        if not self.models_evaluators:
            raise ValueError("No models registered for evaluation")
        
        if self.verbose:
            logger.info(f"Evaluating {len(self.models_evaluators)} models...")
        
        for model_name, evaluator in self.models_evaluators.items():
            if self.verbose:
                logger.info(f"Evaluating: {model_name}")
            
            try:
                results = evaluator.evaluate_batch(emails_data)
                self.models_results[model_name] = results
            except Exception as e:
                logger.error(f"Error evaluating model {model_name}: {str(e)}")
        
        return self.models_results
    
    def get_model_rankings(
        self,
        metric: str = "micro_f1",
        argument_type: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Rank models by a specific metric.
        
        Args:
            metric: Metric to use for ranking (micro_f1, macro_f1, etc.)
            argument_type: Specific argument type (None = overall)
            
        Returns:
            List of (model_name, score) sorted by score descending
        """
        if not self.models_results:
            logger.warning("No evaluation results available")
            return []
        
        rankings = []
        
        for model_name, results in self.models_results.items():
            report = results.get("evaluation_report", {})
            
            if argument_type:
                # Get score for specific argument type
                per_class = report.get("per_class_metrics", {})
                arg_metric = per_class.get(argument_type, {})
                
                # Try different metric names
                if metric == "micro_f1":
                    score = arg_metric.get("f1", 0)
                elif metric == "macro_f1":
                    score = arg_metric.get("f1", 0)
                else:
                    score = arg_metric.get(metric, 0)
            else:
                # Get overall aggregated score
                aggregated = report.get("aggregated_metrics", {})
                
                if metric == "micro_f1":
                    score = aggregated.get("micro_f1", 0)
                elif metric == "macro_f1":
                    score = aggregated.get("macro_f1", 0)
                elif metric == "weighted_f1":
                    score = aggregated.get("weighted_f1", 0)
                else:
                    score = aggregated.get(metric, 0)
            
            rankings.append((model_name, score))
        
        # Sort by score descending
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def compare_models_on_metric(
        self,
        metric: str = "f1",
        argument_type: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Get scores for all models on a specific metric.
        
        Args:
            metric: Metric name
            argument_type: Optional specific argument type
            
        Returns:
            Dictionary mapping model names to scores
        """
        scores = {}
        
        for model_name, results in self.models_results.items():
            report = results.get("evaluation_report", {})
            
            if argument_type:
                per_class = report.get("per_class_metrics", {})
                score = per_class.get(argument_type, {}).get(metric, 0)
            else:
                aggregated = report.get("aggregated_metrics", {})
                score = aggregated.get(metric, 0)
            
            scores[model_name] = score
        
        return scores
    
    def get_best_model(
        self,
        metric: str = "micro_f1"
    ) -> Optional[str]:
        """
        Get the best performing model.
        
        Args:
            metric: Metric to use
            
        Returns:
            Name of best model or None
        """
        rankings = self.get_model_rankings(metric)
        return rankings[0][0] if rankings else None
    
    def get_worst_model(
        self,
        metric: str = "micro_f1"
    ) -> Optional[str]:
        """
        Get the worst performing model.
        
        Args:
            metric: Metric to use
            
        Returns:
            Name of worst model or None
        """
        rankings = self.get_model_rankings(metric)
        return rankings[-1][0] if rankings else None
    
    def compare_on_argument_type(
        self,
        argument_type: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare all models on specific argument type.
        
        Args:
            argument_type: Argument type to compare
            
        Returns:
            Dictionary mapping model names to metrics
        """
        comparison = {}
        
        for model_name, results in self.models_results.items():
            report = results.get("evaluation_report", {})
            per_class = report.get("per_class_metrics", {})
            
            arg_metrics = per_class.get(argument_type, {})
            comparison[model_name] = {
                "precision": arg_metrics.get("precision", 0),
                "recall": arg_metrics.get("recall", 0),
                "f1": arg_metrics.get("f1", 0),
                "support": arg_metrics.get("support", 0)
            }
        
        return comparison
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics across all models.
        
        Returns:
            Dictionary with summary information
        """
        if not self.models_results:
            return {}
        
        model_names = list(self.models_results.keys())
        
        # Collect micro F1 scores
        micro_f1_scores = []
        macro_f1_scores = []
        accuracies = []
        
        for results in self.models_results.values():
            report = results.get("evaluation_report", {})
            aggregated = report.get("aggregated_metrics", {})
            
            micro_f1_scores.append(aggregated.get("micro_f1", 0))
            macro_f1_scores.append(aggregated.get("macro_f1", 0))
            accuracies.append(aggregated.get("accuracy", 0))
        
        summary = {
            "num_models": len(model_names),
            "micro_f1": {
                "best": max(micro_f1_scores),
                "worst": min(micro_f1_scores),
                "mean": sum(micro_f1_scores) / len(micro_f1_scores),
                "median": self._median(micro_f1_scores),
                "std_dev": self._std_dev(micro_f1_scores)
            },
            "macro_f1": {
                "best": max(macro_f1_scores),
                "worst": min(macro_f1_scores),
                "mean": sum(macro_f1_scores) / len(macro_f1_scores),
                "median": self._median(macro_f1_scores),
                "std_dev": self._std_dev(macro_f1_scores)
            },
            "accuracy": {
                "best": max(accuracies),
                "worst": min(accuracies),
                "mean": sum(accuracies) / len(accuracies),
                "median": self._median(accuracies),
                "std_dev": self._std_dev(accuracies)
            }
        }
        
        return summary
    
    @staticmethod
    def _median(values: List[float]) -> float:
        """Calculate median."""
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n % 2 == 0:
            return (sorted_vals[n//2 - 1] + sorted_vals[n//2]) / 2
        else:
            return sorted_vals[n//2]
    
    @staticmethod
    def _std_dev(values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def identify_model_strengths_weaknesses(
        self,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Identify strengths and weaknesses of a model.
        
        Args:
            model_name: Name of model to analyze
            
        Returns:
            Dictionary with strength/weakness analysis
        """
        if model_name not in self.models_results:
            logger.warning(f"Model {model_name} not found")
            return {}
        
        results = self.models_results[model_name]
        report = results.get("evaluation_report", {})
        per_class = report.get("per_class_metrics", {})
        
        # Find best and worst argument types
        arg_f1_scores = {
            arg: metrics.get("f1", 0) 
            for arg, metrics in per_class.items()
        }
        
        best_arg = max(arg_f1_scores, key=arg_f1_scores.get)
        worst_arg = min(arg_f1_scores, key=arg_f1_scores.get)
        
        # Error analysis
        error_analysis = results.get("error_analysis", {})
        by_argument = error_analysis.get("by_argument", {})
        
        analysis = {
            "best_argument_type": {
                "name": best_arg,
                "f1": arg_f1_scores[best_arg]
            },
            "worst_argument_type": {
                "name": worst_arg,
                "f1": arg_f1_scores[worst_arg]
            },
            "error_distribution": {
                arg: by_argument.get(arg, {}).get("count", 0)
                for arg in per_class.keys()
            }
        }
        
        return analysis
    
    def save_comparison_results(
        self,
        output_path: str
    ) -> None:
        """
        Save comparison results to JSON.
        
        Args:
            output_path: Path to save results
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        comparison_data = {
            "timestamp": datetime.now().isoformat(),
            "num_models": len(self.models_results),
            "models": self.models_results,
            "rankings": {
                "micro_f1": self.get_model_rankings("micro_f1"),
                "macro_f1": self.get_model_rankings("macro_f1")
            },
            "summary_statistics": self.get_summary_statistics()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Custom serializer for floats
            json.dump(comparison_data, f, indent=2, ensure_ascii=False, default=str)
        
        if self.verbose:
            logger.info(f"Comparison results saved to {output_path}")
    
    def print_comparison_summary(self) -> None:
        """Print comparison summary to console."""
        if not self.models_results:
            logger.warning("No evaluation results available")
            return
        
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80 + "\n")
        
        # Rankings by Micro F1
        print("Rankings (Micro F1):")
        print("-" * 50)
        rankings = self.get_model_rankings("micro_f1")
        for i, (model_name, score) in enumerate(rankings, 1):
            print(f"{i}. {model_name}: {score:.4f}")
        
        print("\n")
        
        # Summary statistics
        print("Summary Statistics:")
        print("-" * 50)
        summary = self.get_summary_statistics()
        
        print(f"Micro F1:")
        print(f"  Mean: {summary['micro_f1']['mean']:.4f}")
        print(f"  Std Dev: {summary['micro_f1']['std_dev']:.4f}")
        print(f"  Best: {summary['micro_f1']['best']:.4f}")
        print(f"  Worst: {summary['micro_f1']['worst']:.4f}")
        
        print(f"\nMacro F1:")
        print(f"  Mean: {summary['macro_f1']['mean']:.4f}")
        print(f"  Std Dev: {summary['macro_f1']['std_dev']:.4f}")
        
        print("\n" + "="*80 + "\n")
