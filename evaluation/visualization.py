"""
Visualization Module for Argument Extraction Evaluation

Generates plots and visualizations:
- Confusion matrices per argument type
- Bar charts comparing metrics
- Model comparison plots
- F1-score heatmaps
- Error distribution plots

Uses matplotlib for maximum compatibility.
Optionally uses seaborn for enhanced aesthetics.

Author: Automatic Evaluation Framework
License: MIT
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Generates visualizations from evaluation results.
    
    Supports:
    - Per-model visualizations
    - Comparative visualizations
    - Multiple output formats
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 300,
        style: str = "seaborn" if HAS_SEABORN else "default",
        font_size: int = 10
    ):
        """
        Initialize visualizer.
        
        Args:
            figsize: Figure size (width, height)
            dpi: Resolution
            style: Plotting style
            font_size: Font size for labels
        """
        self.figsize = figsize
        self.dpi = dpi
        self.font_size = font_size
        
        if style == "seaborn" and HAS_SEABORN:
            sns.set_style("whitegrid")
            self.style = "seaborn"
        else:
            self.style = "default"
        
        plt.rcParams['font.size'] = font_size
        plt.rcParams['figure.figsize'] = figsize
        plt.rcParams['figure.dpi'] = dpi
    
    def plot_per_argument_metrics(
        self,
        evaluation_data: Dict,
        output_path: str,
        metric: str = "f1"
    ) -> None:
        """
        Plot metrics for each argument type as bar chart.
        
        Args:
            evaluation_data: Evaluation results
            output_path: Path to save figure
            metric: Metric to plot ("precision", "recall", or "f1")
        """
        report = evaluation_data.get("evaluation_report", {})
        per_class = report.get("per_class_metrics", {})
        
        if not per_class:
            logger.warning("No per-class metrics found")
            return
        
        argument_types = list(per_class.keys())
        metrics_values = [per_class[arg].get(metric, 0) for arg in argument_types]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(argument_types)))
        bars = ax.bar(argument_types, metrics_values, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}',
                   ha='center', va='bottom', fontsize=self.font_size)
        
        ax.set_ylabel(metric.capitalize(), fontsize=self.font_size + 2, fontweight='bold')
        ax.set_xlabel('Argument Type', fontsize=self.font_size + 2, fontweight='bold')
        ax.set_title(f'{metric.capitalize()} Score by Argument Type', 
                    fontsize=self.font_size + 4, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(output_path)
    
    def plot_confusion_matrix(
        self,
        evaluation_data: Dict,
        argument_type: str,
        output_path: str
    ) -> None:
        """
        Plot confusion matrix for specific argument type.
        
        Args:
            evaluation_data: Evaluation results
            argument_type: Argument type to visualize
            output_path: Path to save figure
        """
        report = evaluation_data.get("evaluation_report", {})
        per_class = report.get("per_class_metrics", {})
        
        if argument_type not in per_class:
            logger.warning(f"Argument type {argument_type} not found")
            return
        
        confusion = per_class[argument_type].get("confusion", {})
        
        # Create confusion matrix
        tp = confusion.get("tp", 0)
        fp = confusion.get("fp", 0)
        fn = confusion.get("fn", 0)
        tn = confusion.get("tn", 0)
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        
        # Labels
        labels = ['Not Present', 'Present']
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        ax.set_ylabel('Actual', fontsize=self.font_size + 2, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=self.font_size + 2, fontweight='bold')
        ax.set_title(f'Confusion Matrix: {argument_type}',
                    fontsize=self.font_size + 4, fontweight='bold', pad=20)
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, cm[i, j],
                             ha="center", va="center", color="black",
                             fontsize=self.font_size + 4, fontweight='bold')
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        self._save_figure(output_path)
    
    def plot_precision_recall_f1(
        self,
        evaluation_data: Dict,
        output_path: str
    ) -> None:
        """
        Plot precision, recall, and F1 for all argument types.
        
        Args:
            evaluation_data: Evaluation results
            output_path: Path to save figure
        """
        report = evaluation_data.get("evaluation_report", {})
        per_class = report.get("per_class_metrics", {})
        
        if not per_class:
            logger.warning("No per-class metrics found")
            return
        
        argument_types = list(per_class.keys())
        precisions = [per_class[arg].get("precision", 0) for arg in argument_types]
        recalls = [per_class[arg].get("recall", 0) for arg in argument_types]
        f1s = [per_class[arg].get("f1", 0) for arg in argument_types]
        
        x = np.arange(len(argument_types))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        rects1 = ax.bar(x - width, precisions, width, label='Precision', 
                       color='#FF6B6B', edgecolor='black', linewidth=1.5)
        rects2 = ax.bar(x, recalls, width, label='Recall', 
                       color='#4ECDC4', edgecolor='black', linewidth=1.5)
        rects3 = ax.bar(x + width, f1s, width, label='F1', 
                       color='#45B7D1', edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Score', fontsize=self.font_size + 2, fontweight='bold')
        ax.set_xlabel('Argument Type', fontsize=self.font_size + 2, fontweight='bold')
        ax.set_title('Precision, Recall, and F1-Score by Argument Type',
                    fontsize=self.font_size + 4, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(argument_types)
        ax.legend(fontsize=self.font_size)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for rects in [rects1, rects2, rects3]:
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        self._save_figure(output_path)
    
    def plot_model_comparison(
        self,
        models_results: Dict[str, Dict],
        output_path: str,
        metric: str = "f1",
        argument_type: Optional[str] = None
    ) -> None:
        """
        Plot comparison of multiple models.
        
        Args:
            models_results: Dictionary mapping model names to results
            output_path: Path to save figure
            metric: Metric to compare ("precision", "recall", "f1")
            argument_type: Specific argument type (None = averaged)
        """
        model_names = list(models_results.keys())
        model_scores = []
        
        for model_name, results in models_results.items():
            report = results.get("evaluation_report", {})
            per_class = report.get("per_class_metrics", {})
            
            if argument_type and argument_type in per_class:
                score = per_class[argument_type].get(metric, 0)
            else:
                # Average across all argument types
                scores = [per_class.get(arg, {}).get(metric, 0) 
                         for arg in ["participants", "time", "location", "topic"]]
                score = np.mean(scores) if scores else 0
            
            model_scores.append(score)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = plt.cm.Spectral(np.linspace(0, 1, len(model_names)))
        bars = ax.bar(model_names, model_scores, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, score in zip(bars, model_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}',
                   ha='center', va='bottom', fontsize=self.font_size)
        
        title = f'Model Comparison: {metric.capitalize()}'
        if argument_type:
            title += f' ({argument_type})'
        
        ax.set_ylabel(metric.capitalize(), fontsize=self.font_size + 2, fontweight='bold')
        ax.set_xlabel('Model', fontsize=self.font_size + 2, fontweight='bold')
        ax.set_title(title, fontsize=self.font_size + 4, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        self._save_figure(output_path)
    
    def plot_error_distribution(
        self,
        evaluation_data: Dict,
        output_path: str
    ) -> None:
        """
        Plot distribution of error types.
        
        Args:
            evaluation_data: Evaluation results
            output_path: Path to save figure
        """
        summary = evaluation_data.get("summary_statistics", {})
        
        error_types = ['False Positives', 'False Negatives', 'Partial Matches']
        error_counts = [
            summary.get('false_positives', 0),
            summary.get('false_negatives', 0),
            summary.get('partial_matches', 0)
        ]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Pie chart
        colors = ['#FF6B6B', '#4ECDC4', '#FFE66D']
        wedges, texts, autotexts = ax1.pie(error_counts, labels=error_types, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(self.font_size)
        
        ax1.set_title('Error Type Distribution', 
                     fontsize=self.font_size + 2, fontweight='bold')
        
        # Bar chart
        ax2.bar(error_types, error_counts, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Count', fontsize=self.font_size + 2, fontweight='bold')
        ax2.set_title('Error Type Counts',
                     fontsize=self.font_size + 2, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (error_type, count) in enumerate(zip(error_types, error_counts)):
            ax2.text(i, count, str(count), ha='center', va='bottom', 
                    fontsize=self.font_size, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure(output_path)
    
    def plot_error_by_argument(
        self,
        evaluation_data: Dict,
        output_path: str
    ) -> None:
        """
        Plot error counts per argument type.
        
        Args:
            evaluation_data: Evaluation results
            output_path: Path to save figure
        """
        error_analysis = evaluation_data.get("error_analysis", {})
        by_argument = error_analysis.get("by_argument", {})
        
        argument_types = list(by_argument.keys())
        error_counts = [by_argument[arg].get('count', 0) for arg in argument_types]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(argument_types)))
        bars = ax.bar(argument_types, error_counts, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, count in zip(bars, error_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   str(int(count)),
                   ha='center', va='bottom', fontsize=self.font_size, fontweight='bold')
        
        ax.set_ylabel('Error Count', fontsize=self.font_size + 2, fontweight='bold')
        ax.set_xlabel('Argument Type', fontsize=self.font_size + 2, fontweight='bold')
        ax.set_title('Error Distribution by Argument Type',
                    fontsize=self.font_size + 4, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(output_path)
    
    def plot_f1_heatmap(
        self,
        models_results: Dict[str, Dict],
        output_path: str
    ) -> None:
        """
        Plot F1 scores as heatmap (models vs argument types).
        
        Args:
            models_results: Dictionary mapping model names to results
            output_path: Path to save figure
        """
        if not HAS_SEABORN:
            logger.warning("Seaborn not available for heatmap. Using alternative visualization.")
            self._plot_f1_matrix(models_results, output_path)
            return
        
        model_names = list(models_results.keys())
        argument_types = ["participants", "time", "location", "topic"]
        
        data = np.zeros((len(argument_types), len(model_names)))
        
        for j, model_name in enumerate(model_names):
            results = models_results[model_name]
            report = results.get("evaluation_report", {})
            per_class = report.get("per_class_metrics", {})
            
            for i, arg_type in enumerate(argument_types):
                f1 = per_class.get(arg_type, {}).get("f1", 0)
                data[i, j] = f1
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        im = sns.heatmap(data, annot=True, fmt='.3f', cmap='YlGnBu',
                        xticklabels=model_names, yticklabels=argument_types,
                        cbar_kws={'label': 'F1 Score'}, ax=ax, vmin=0, vmax=1)
        
        ax.set_title('F1-Score Heatmap: Models vs Argument Types',
                    fontsize=self.font_size + 4, fontweight='bold', pad=20)
        ax.set_xlabel('Model', fontsize=self.font_size + 2, fontweight='bold')
        ax.set_ylabel('Argument Type', fontsize=self.font_size + 2, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure(output_path)
    
    def _plot_f1_matrix(
        self,
        models_results: Dict[str, Dict],
        output_path: str
    ) -> None:
        """Alternative F1 matrix plot without seaborn."""
        model_names = list(models_results.keys())
        argument_types = ["participants", "time", "location", "topic"]
        
        f1_data = {}
        for model_name in model_names:
            results = models_results[model_name]
            report = results.get("evaluation_report", {})
            per_class = report.get("per_class_metrics", {})
            f1_data[model_name] = [per_class.get(arg, {}).get("f1", 0) for arg in argument_types]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = np.arange(len(argument_types))
        width = 0.8 / len(model_names)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        
        for i, (model_name, f1_scores) in enumerate(f1_data.items()):
            offset = (i - len(model_names)/2) * width + width/2
            ax.bar(x + offset, f1_scores, width, label=model_name, 
                  color=colors[i], edgecolor='black', linewidth=1)
        
        ax.set_ylabel('F1 Score', fontsize=self.font_size + 2, fontweight='bold')
        ax.set_xlabel('Argument Type', fontsize=self.font_size + 2, fontweight='bold')
        ax.set_title('F1-Score Comparison: Models vs Argument Types',
                    fontsize=self.font_size + 4, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(argument_types)
        ax.legend(fontsize=self.font_size)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(output_path)
    
    def _save_figure(self, output_path: str) -> None:
        """Save figure to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Figure saved to {output_path}")
    
    def generate_all_visualizations(
        self,
        evaluation_data: Dict,
        output_dir: str
    ) -> None:
        """
        Generate all standard visualizations.
        
        Args:
            evaluation_data: Evaluation results
            output_dir: Directory to save visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating visualizations in {output_dir}...")
        
        # Generate all plots
        self.plot_per_argument_metrics(evaluation_data, 
                                      str(output_path / "metrics_f1.png"))
        self.plot_precision_recall_f1(evaluation_data,
                                     str(output_path / "precision_recall_f1.png"))
        self.plot_error_distribution(evaluation_data,
                                    str(output_path / "error_distribution.png"))
        self.plot_error_by_argument(evaluation_data,
                                   str(output_path / "error_by_argument.png"))
        
        # Per-argument confusion matrices
        for arg_type in ["participants", "time", "location", "topic"]:
            self.plot_confusion_matrix(evaluation_data, arg_type,
                                      str(output_path / f"confusion_matrix_{arg_type}.png"))
        
        logger.info("All visualizations generated successfully")
