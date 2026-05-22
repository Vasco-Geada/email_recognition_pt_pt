"""
Report Generator for Argument Extraction Evaluation

Generates comprehensive evaluation reports in multiple formats:
- JSON (detailed)
- CSV (tabular)
- Markdown (human-readable)
- LaTeX tables (for dissertations)

Features:
- Per-model reports
- Comparative reports
- Model ranking
- Argument type analysis

Author: Automatic Evaluation Framework
License: MIT
"""

import json
import csv
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
import logging


logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates reports from evaluation results.
    
    Supports multiple output formats and organizational structures.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize report generator.
        
        Args:
            verbose: Print progress information
        """
        self.verbose = verbose
    
    def generate_json_report(
        self,
        evaluation_data: Dict[str, Any],
        output_path: str,
        pretty: bool = True
    ) -> None:
        """
        Generate JSON report.
        
        Args:
            evaluation_data: Evaluation results dictionary
            output_path: Path to save report
            pretty: Pretty-print JSON
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(
                evaluation_data,
                f,
                indent=2 if pretty else None,
                ensure_ascii=False
            )
        
        if self.verbose:
            logger.info(f"JSON report saved to {output_path}")
    
    def generate_csv_report(
        self,
        evaluation_data: Dict[str, Any],
        output_path: str,
        report_type: str = "metrics"
    ) -> None:
        """
        Generate CSV report.
        
        Args:
            evaluation_data: Evaluation results dictionary
            output_path: Path to save report
            report_type: Type of report ("metrics", "errors", or "emails")
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if report_type == "metrics":
            self._generate_metrics_csv(evaluation_data, output_file)
        elif report_type == "errors":
            self._generate_errors_csv(evaluation_data, output_file)
        elif report_type == "emails":
            self._generate_emails_csv(evaluation_data, output_file)
        else:
            logger.warning(f"Unknown report type: {report_type}")
        
        if self.verbose:
            logger.info(f"CSV report ({report_type}) saved to {output_path}")
    
    def _generate_metrics_csv(self, evaluation_data: Dict, output_file: Path) -> None:
        """Generate metrics CSV."""
        report = evaluation_data.get("evaluation_report", {})
        per_class = report.get("per_class_metrics", {})
        aggregated = report.get("aggregated_metrics", {})
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(["Argument Type", "Precision", "Recall", "F1", "Support", "TP", "FP", "FN"])
            
            # Per-class metrics
            for arg_type, metrics in per_class.items():
                writer.writerow([
                    arg_type,
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['f1']:.4f}",
                    metrics['support'],
                    metrics['confusion']['tp'],
                    metrics['confusion']['fp'],
                    metrics['confusion']['fn']
                ])
            
            # Empty row
            writer.writerow([])
            
            # Aggregated metrics
            writer.writerow(["Aggregation Method", "Precision", "Recall", "F1"])
            
            micro = aggregated.get("micro", {})
            writer.writerow([
                "Micro Average",
                f"{micro['precision']:.4f}",
                f"{micro['recall']:.4f}",
                f"{micro['f1']:.4f}"
            ])
            
            macro = aggregated.get("macro", {})
            writer.writerow([
                "Macro Average",
                f"{macro['precision']:.4f}",
                f"{macro['recall']:.4f}",
                f"{macro['f1']:.4f}"
            ])
            
            writer.writerow(["Weighted F1", "", "", f"{aggregated.get('weighted_f1', 0):.4f}"])
            writer.writerow(["Accuracy", "", "", f"{aggregated.get('accuracy', 0):.4f}"])
    
    def _generate_errors_csv(self, evaluation_data: Dict, output_file: Path) -> None:
        """Generate errors CSV."""
        error_analysis = evaluation_data.get("error_analysis", {})
        errors = error_analysis.get("by_type", {}).get("false_negative", [])
        
        if not errors:
            errors = error_analysis.get("by_type", {}).get("false_positive", [])
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(["Email ID", "Argument Type", "Error Type", "Gold Value", "Predicted Value"])
            
            # Error rows
            for error in errors[:100]:  # Limit to 100 for readability
                writer.writerow([
                    error.get("email_id", ""),
                    error.get("argument_type", ""),
                    error.get("error_type", ""),
                    error.get("gold_value", ""),
                    error.get("predicted_value", "")
                ])
    
    def _generate_emails_csv(self, evaluation_data: Dict, output_file: Path) -> None:
        """Generate per-email metrics CSV."""
        email_results = evaluation_data.get("email_results", [])
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            header = ["Email ID", "Intent"]
            header.extend([f"{arg}_{metric}" for arg in ["participants", "time", "location", "topic"] 
                          for metric in ["Precision", "Recall", "F1"]])
            writer.writerow(header)
            
            # Email rows
            for result in email_results:
                row = [result.get("email_id", ""), result.get("intent", "")]
                
                metrics = result.get("per_argument_metrics", {})
                for arg_type in ["participants", "time", "location", "topic"]:
                    arg_metrics = metrics.get(arg_type, {})
                    row.extend([
                        f"{arg_metrics.get('precision', 0):.4f}",
                        f"{arg_metrics.get('recall', 0):.4f}",
                        f"{arg_metrics.get('f1', 0):.4f}"
                    ])
                
                writer.writerow(row)
    
    def generate_markdown_report(
        self,
        evaluation_data: Dict[str, Any],
        output_path: str,
        model_name: str = "Argument Extraction Model"
    ) -> None:
        """
        Generate Markdown report (human-readable).
        
        Args:
            evaluation_data: Evaluation results dictionary
            output_path: Path to save report
            model_name: Name of the model
        """
        report = evaluation_data.get("evaluation_report", {})
        per_class = report.get("per_class_metrics", {})
        aggregated = report.get("aggregated_metrics", {})
        summary = evaluation_data.get("summary_statistics", {})
        error_analysis = evaluation_data.get("error_analysis", {})
        
        lines = []
        
        # Header
        lines.append(f"# Evaluation Report: {model_name}\n")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Summary statistics
        lines.append("## Summary Statistics\n")
        lines.append(f"- Total emails evaluated: {summary.get('total_emails', 0)}")
        lines.append(f"- Total errors: {summary.get('total_errors', 0)}")
        lines.append(f"- False positives: {summary.get('false_positives', 0)}")
        lines.append(f"- False negatives: {summary.get('false_negatives', 0)}")
        lines.append(f"- Partial matches: {summary.get('partial_matches', 0)}\n")
        
        # Per-class metrics table
        lines.append("## Per-Argument-Type Metrics\n")
        lines.append("| Argument Type | Precision | Recall | F1 | Support |")
        lines.append("|---------------|-----------|--------|-----|---------|")
        
        for arg_type, metrics in per_class.items():
            lines.append(
                f"| {arg_type} | "
                f"{metrics['precision']:.4f} | "
                f"{metrics['recall']:.4f} | "
                f"{metrics['f1']:.4f} | "
                f"{metrics['support']} |"
            )
        lines.append("")
        
        # Aggregated metrics
        lines.append("## Aggregated Metrics\n")
        
        micro = aggregated.get("micro", {})
        macro = aggregated.get("macro", {})
        
        lines.append("### Micro Average")
        lines.append(f"- Precision: {micro.get('precision', 0):.4f}")
        lines.append(f"- Recall: {micro.get('recall', 0):.4f}")
        lines.append(f"- F1: {micro.get('f1', 0):.4f}\n")
        
        lines.append("### Macro Average")
        lines.append(f"- Precision: {macro.get('precision', 0):.4f}")
        lines.append(f"- Recall: {macro.get('recall', 0):.4f}")
        lines.append(f"- F1: {macro.get('f1', 0):.4f}\n")
        
        lines.append(f"### Overall Metrics")
        lines.append(f"- Weighted F1: {aggregated.get('weighted_f1', 0):.4f}")
        lines.append(f"- Accuracy: {aggregated.get('accuracy', 0):.4f}\n")
        
        # Error analysis
        lines.append("## Error Analysis\n")
        
        by_arg = error_analysis.get("by_argument", {})
        top_errors = error_analysis.get("top_error_arguments", [])
        
        if top_errors:
            lines.append("### Most Common Error Arguments\n")
            for arg_type, counts_dict in top_errors:
                lines.append(f"- **{arg_type}**: {counts_dict.get('count', 0)} errors")
            lines.append("")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        if self.verbose:
            logger.info(f"Markdown report saved to {output_path}")
    
    def generate_latex_table(
        self,
        evaluation_data: Dict[str, Any],
        output_path: str
    ) -> None:
        """
        Generate LaTeX table for dissertation.
        
        Args:
            evaluation_data: Evaluation results dictionary
            output_path: Path to save table
        """
        report = evaluation_data.get("evaluation_report", {})
        per_class = report.get("per_class_metrics", {})
        aggregated = report.get("aggregated_metrics", {})
        
        lines = []
        
        # LaTeX table header
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{|l|r|r|r|r|}")
        lines.append(r"\hline")
        lines.append(r"\textbf{Argument Type} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{Support} \\")
        lines.append(r"\hline")
        
        # Per-class rows
        for arg_type, metrics in per_class.items():
            lines.append(
                f"{arg_type} & "
                f"{metrics['precision']:.4f} & "
                f"{metrics['recall']:.4f} & "
                f"{metrics['f1']:.4f} & "
                f"{metrics['support']} \\\\"
            )
        
        lines.append(r"\hline")
        
        # Aggregated rows
        micro = aggregated.get("micro", {})
        macro = aggregated.get("macro", {})
        
        lines.append(
            f"\\textbf{{Micro Avg}} & "
            f"{micro.get('precision', 0):.4f} & "
            f"{micro.get('recall', 0):.4f} & "
            f"{micro.get('f1', 0):.4f} & - \\\\"
        )
        
        lines.append(
            f"\\textbf{{Macro Avg}} & "
            f"{macro.get('precision', 0):.4f} & "
            f"{macro.get('recall', 0):.4f} & "
            f"{macro.get('f1', 0):.4f} & - \\\\"
        )
        
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Argument Extraction Evaluation Metrics}")
        lines.append(r"\label{tab:arg_extraction_metrics}")
        lines.append(r"\end{table}")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        if self.verbose:
            logger.info(f"LaTeX table saved to {output_path}")
    
    def generate_comparative_report(
        self,
        models_results: Dict[str, Dict[str, Any]],
        output_path: str,
        output_format: str = "markdown"
    ) -> None:
        """
        Generate comparative report for multiple models.
        
        Args:
            models_results: Dictionary mapping model names to evaluation results
            output_path: Path to save report
            output_format: Output format ("markdown", "csv", "json")
        """
        if output_format == "markdown":
            self._generate_comparative_markdown(models_results, output_path)
        elif output_format == "csv":
            self._generate_comparative_csv(models_results, output_path)
        elif output_format == "json":
            with open(Path(output_path), 'w', encoding='utf-8') as f:
                json.dump(models_results, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            logger.info(f"Comparative report ({output_format}) saved to {output_path}")
    
    def _generate_comparative_markdown(
        self,
        models_results: Dict[str, Dict[str, Any]],
        output_path: str
    ) -> None:
        """Generate comparative Markdown report."""
        lines = []
        
        lines.append("# Model Comparison Report\n")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        lines.append("## Aggregated Metrics Comparison\n")
        lines.append("| Model | Micro F1 | Macro F1 | Weighted F1 | Accuracy |")
        lines.append("|-------|----------|----------|-------------|----------|")
        
        for model_name, results in models_results.items():
            report = results.get("evaluation_report", {})
            aggregated = report.get("aggregated_metrics", {})
            
            lines.append(
                f"| {model_name} | "
                f"{aggregated.get('micro_f1', 0):.4f} | "
                f"{aggregated.get('macro_f1', 0):.4f} | "
                f"{aggregated.get('weighted_f1', 0):.4f} | "
                f"{aggregated.get('accuracy', 0):.4f} |"
            )
        
        lines.append("")
        
        # Per-argument comparison
        lines.append("## Per-Argument-Type Comparison (F1 Scores)\n")
        lines.append("| Argument Type |")
        lines.extend([f" {name} |" for name in models_results.keys()])
        lines[0] = "| Argument Type |" + ''.join([f" {name} |" for name in models_results.keys()])
        
        lines.append("|" + "".join(["-" * 12 for _ in models_results.keys()]) + "----|")
        
        for arg_type in ["participants", "time", "location", "topic"]:
            row = f"| {arg_type} |"
            for model_name, results in models_results.items():
                report = results.get("evaluation_report", {})
                per_class = report.get("per_class_metrics", {})
                f1 = per_class.get(arg_type, {}).get("f1", 0)
                row += f" {f1:.4f} |"
            lines.append(row)
        
        lines.append("")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    def _generate_comparative_csv(
        self,
        models_results: Dict[str, Dict[str, Any]],
        output_path: str
    ) -> None:
        """Generate comparative CSV report."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            header = ["Metric"] + list(models_results.keys())
            writer.writerow(header)
            
            # Metrics rows
            metrics_to_compare = [
                ("Micro F1", lambda r: r.get("aggregated_metrics", {}).get("micro_f1", 0)),
                ("Macro F1", lambda r: r.get("aggregated_metrics", {}).get("macro_f1", 0)),
                ("Weighted F1", lambda r: r.get("aggregated_metrics", {}).get("weighted_f1", 0)),
                ("Accuracy", lambda r: r.get("aggregated_metrics", {}).get("accuracy", 0))
            ]
            
            for metric_name, extractor in metrics_to_compare:
                row = [metric_name]
                for model_name, results in models_results.items():
                    report = results.get("evaluation_report", {})
                    value = extractor(report)
                    row.append(f"{value:.4f}")
                writer.writerow(row)
