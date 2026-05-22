"""
Complete Example: Evaluating Argument Extraction Models

This script demonstrates a complete workflow for:
1. Loading gold annotations and predictions
2. Evaluating models individually
3. Comparing multiple models
4. Generating reports and visualizations
5. Error analysis

Usage:
    python example_usage.py

Author: Automatic Evaluation Framework
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

# Import evaluation framework
from evaluation import (
    ArgumentExtractionEvaluator,
    ModelComparator,
    ReportGenerator,
    Visualizer,
    DataLoader,
    DataValidator,
    DataPreprocessor,
    EvaluationUtils
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data() -> tuple[List[Dict], Dict[str, List[Dict]]]:
    """
    Create sample evaluation data for demonstration.
    
    Returns:
        Tuple of (gold_annotations, predictions_by_model)
    """
    
    # Sample gold annotations
    gold_annotations = [
        {
            "id": 1,
            "text": "Boas Ana, podemos reunir amanhã às 15h no Teams?",
            "intent": "agendamento_reuniao",
            "arguments": {
                "participants": ["Ana"],
                "time": ["amanhã às 15h"],
                "location": ["Teams"],
                "topic": []
            }
        },
        {
            "id": 2,
            "text": "Infelizmente tenho que cancelar a reunião de sexta às 14h.",
            "intent": "cancelamento_reuniao",
            "arguments": {
                "participants": [],
                "time": ["sexta às 14h"],
                "location": [],
                "topic": []
            }
        },
        {
            "id": 3,
            "text": "Confirmado o encontro com o João na próxima segunda de manhã.",
            "intent": "reuniao_confirmada",
            "arguments": {
                "participants": ["João"],
                "time": ["próxima segunda de manhã"],
                "location": [],
                "topic": []
            }
        },
        {
            "id": 4,
            "text": "Podemos marcar para terça? Talvez no escritório.",
            "intent": "agendamento_reuniao",
            "arguments": {
                "participants": [],
                "time": ["terça"],
                "location": ["escritório"],
                "topic": []
            }
        }
    ]
    
    # Sample predictions from different models
    predictions_regex = {
        1: {
            "participants": ["Ana"],
            "time": ["amanhã às 15h"],
            "location": ["Teams"],
            "topic": []
        },
        2: {
            "participants": [],
            "time": ["sexta", "14h"],  # Partial match - split
            "location": [],
            "topic": []
        },
        3: {
            "participants": ["João"],
            "time": ["próxima segunda"],  # Partial match - missing "de manhã"
            "location": [],
            "topic": []
        },
        4: {
            "participants": [],
            "time": ["terça"],
            "location": [],
            "topic": []
        }
    }
    
    predictions_spacy = {
        1: {
            "participants": ["Ana"],
            "time": ["15h"],  # False negative - missing "amanhã às"
            "location": ["Teams"],
            "topic": []
        },
        2: {
            "participants": [],
            "time": ["sexta às 14h"],
            "location": [],
            "topic": []
        },
        3: {
            "participants": ["João"],
            "time": ["segunda"],  # False negative
            "location": [],
            "topic": ["reunião"]  # False positive
        },
        4: {
            "participants": [],
            "time": [],  # False negative
            "location": ["escritório"],
            "topic": []
        }
    }
    
    predictions_qa = {
        1: {
            "participants": ["Ana"],
            "time": ["amanhã às 15h"],
            "location": ["Teams"],
            "topic": []
        },
        2: {
            "participants": [],
            "time": ["sexta às 14h"],
            "location": [],
            "topic": []
        },
        3: {
            "participants": ["João"],
            "time": ["próxima segunda de manhã"],
            "location": [],
            "topic": []
        },
        4: {
            "participants": ["utilizador"],  # False positive
            "time": ["terça"],
            "location": ["escritório"],
            "topic": []
        }
    }
    
    predictions_by_model = {
        "regex": predictions_regex,
        "spacy": predictions_spacy,
        "qa": predictions_qa
    }
    
    return gold_annotations, predictions_by_model


def merge_data(gold: List[Dict], predictions: Dict) -> List[Dict]:
    """Merge gold annotations with predictions."""
    merged = []
    
    for ann in gold:
        email_id = ann["id"]
        if email_id in predictions:
            item = {
                "id": email_id,
                "text": ann["text"],
                "intent": ann["intent"],
                "arguments": ann["arguments"],
                "predicted": predictions[email_id]
            }
            merged.append(item)
    
    return merged


def evaluate_single_model(
    model_name: str,
    emails_data: List[Dict],
    output_dir: str
) -> Dict:
    """
    Evaluate a single model.
    
    Args:
        model_name: Name of the model
        emails_data: Merged email data
        output_dir: Output directory
        
    Returns:
        Evaluation results
    """
    logger.info(f"Evaluating model: {model_name}")
    
    # Create evaluator
    evaluator = ArgumentExtractionEvaluator(
        exact_match_threshold=1.0,
        partial_match_threshold=0.7,
        fuzzy_match_threshold=0.6,
        normalize_text=True,
        verbose=True
    )
    
    # Evaluate batch
    results = evaluator.evaluate_batch(emails_data)
    
    # Save results
    model_output_dir = Path(output_dir) / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator.save_results(str(model_output_dir / "results.json"))
    evaluator.save_errors(str(model_output_dir / "errors.json"))
    
    logger.info(f"Model {model_name} saved to {model_output_dir}")
    
    return results


def compare_models(
    models_results: Dict[str, Dict],
    output_dir: str
) -> None:
    """
    Compare multiple models.
    
    Args:
        models_results: Results from each model
        output_dir: Output directory
    """
    logger.info("Comparing models...")
    
    # Create comparator with results
    comparator = ModelComparator(verbose=True)
    comparator.models_results = models_results
    
    # Print summary
    comparator.print_comparison_summary()
    
    # Save comparison results
    output_path = Path(output_dir) / "comparison" / "comparison_results.json"
    comparator.save_comparison_results(str(output_path))
    
    logger.info(f"Comparison results saved to {output_path}")


def generate_reports(
    models_results: Dict[str, Dict],
    output_dir: str
) -> None:
    """
    Generate reports for all models.
    
    Args:
        models_results: Results from each model
        output_dir: Output directory
    """
    logger.info("Generating reports...")
    
    report_gen = ReportGenerator(verbose=True)
    
    # Generate per-model reports
    for model_name, results in models_results.items():
        model_output_dir = Path(output_dir) / model_name
        
        # JSON report
        report_gen.generate_json_report(
            results,
            str(model_output_dir / "report.json")
        )
        
        # CSV reports
        report_gen.generate_csv_report(
            results,
            str(model_output_dir / "metrics.csv"),
            report_type="metrics"
        )
        
        report_gen.generate_csv_report(
            results,
            str(model_output_dir / "errors.csv"),
            report_type="errors"
        )
        
        report_gen.generate_csv_report(
            results,
            str(model_output_dir / "emails.csv"),
            report_type="emails"
        )
        
        # Markdown report
        report_gen.generate_markdown_report(
            results,
            str(model_output_dir / "report.md"),
            model_name=model_name
        )
        
        # LaTeX table
        report_gen.generate_latex_table(
            results,
            str(model_output_dir / "table.tex")
        )
    
    # Comparative report
    comparison_dir = Path(output_dir) / "comparison"
    report_gen.generate_comparative_report(
        models_results,
        str(comparison_dir / "comparison.md"),
        output_format="markdown"
    )
    
    report_gen.generate_comparative_report(
        models_results,
        str(comparison_dir / "comparison.csv"),
        output_format="csv"
    )
    
    logger.info("Reports generated successfully")


def generate_visualizations(
    models_results: Dict[str, Dict],
    output_dir: str
) -> None:
    """
    Generate visualizations for all models.
    
    Args:
        models_results: Results from each model
        output_dir: Output directory
    """
    logger.info("Generating visualizations...")
    
    visualizer = Visualizer(figsize=(12, 8), dpi=300)
    
    # Per-model visualizations
    for model_name, results in models_results.items():
        model_viz_dir = Path(output_dir) / model_name / "visualizations"
        model_viz_dir.mkdir(parents=True, exist_ok=True)
        
        visualizer.generate_all_visualizations(
            results,
            str(model_viz_dir)
        )
    
    # Comparative visualizations
    comparison_viz_dir = Path(output_dir) / "comparison" / "visualizations"
    comparison_viz_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer.plot_model_comparison(
        models_results,
        str(comparison_viz_dir / "f1_comparison.png"),
        metric="f1"
    )
    
    visualizer.plot_f1_heatmap(
        models_results,
        str(comparison_viz_dir / "f1_heatmap.png")
    )
    
    logger.info("Visualizations generated successfully")


def main():
    """Run complete evaluation workflow."""
    
    logger.info("="*80)
    logger.info("ARGUMENT EXTRACTION EVALUATION FRAMEWORK - COMPLETE EXAMPLE")
    logger.info("="*80)
    
    # Setup
    output_dir = Path("evaluation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # 1. Create sample data
    logger.info("\n1. Creating sample data...")
    gold_annotations, predictions_by_model = create_sample_data()
    logger.info(f"   Created {len(gold_annotations)} gold annotations")
    logger.info(f"   Created predictions for {len(predictions_by_model)} models")
    
    # 2. Prepare data
    logger.info("\n2. Preparing data...")
    all_models_results = {}
    
    for model_name, predictions in predictions_by_model.items():
        # Merge gold with predictions
        emails_data = merge_data(gold_annotations, predictions)
        
        # Evaluate model
        results = evaluate_single_model(model_name, emails_data, str(output_dir))
        all_models_results[model_name] = results
    
    # 3. Compare models
    logger.info("\n3. Comparing models...")
    compare_models(all_models_results, str(output_dir))
    
    # 4. Generate reports
    logger.info("\n4. Generating reports...")
    generate_reports(all_models_results, str(output_dir))
    
    # 5. Generate visualizations
    logger.info("\n5. Generating visualizations...")
    try:
        generate_visualizations(all_models_results, str(output_dir))
    except Exception as e:
        logger.warning(f"Visualization generation failed: {str(e)}")
    
    # 6. Print summary
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETED")
    logger.info("="*80)
    logger.info(f"\nResults saved to: {output_dir.absolute()}")
    logger.info("\nOutput structure:")
    logger.info("  - regex/")
    logger.info("      - results.json")
    logger.info("      - errors.json")
    logger.info("      - report.json")
    logger.info("      - report.md")
    logger.info("      - *.csv")
    logger.info("      - table.tex")
    logger.info("      - visualizations/")
    logger.info("  - spacy/")
    logger.info("      - [same structure as regex]")
    logger.info("  - qa/")
    logger.info("      - [same structure as regex]")
    logger.info("  - comparison/")
    logger.info("      - comparison.md")
    logger.info("      - comparison.csv")
    logger.info("      - comparison_results.json")
    logger.info("      - visualizations/")
    logger.info("          - f1_comparison.png")
    logger.info("          - f1_heatmap.png")
    
    logger.info("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
