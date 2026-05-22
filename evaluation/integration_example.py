"""
Integration Example: Using Real Project Data

This example shows how to integrate the evaluation framework
with your existing gold annotations and models.

Assuming you have:
- gold_annotations/output/gold.json (gold annotations)
- Predictions from your models
"""

import json
import logging
from pathlib import Path

from evaluation import (
    ArgumentExtractionEvaluator,
    ModelComparator,
    ReportGenerator,
    Visualizer,
    DataLoader,
    DataValidator,
    DataPreprocessor,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_project_gold_annotations(gold_file: str) -> list:
    """
    Load gold annotations from your project.
    
    Assumes gold.json format with:
    - id
    - text
    - intent
    - arguments (participants, time, location, topic)
    """
    logger.info(f"Loading gold annotations from {gold_file}")
    
    gold = DataLoader.load_gold_annotations(gold_file)
    
    # Validate
    try:
        DataValidator.validate_gold_annotations(gold)
        logger.info(f"✓ Validated {len(gold)} gold annotations")
    except ValueError as e:
        logger.error(f"✗ Validation error: {e}")
        return []
    
    # Clean
    gold_clean = [DataPreprocessor.clean_email_data({"arguments": ann["arguments"]}) 
                  for ann in gold]
    for ann_orig, ann_clean in zip(gold, gold_clean):
        ann_orig["arguments"] = ann_clean["arguments"]
    
    return gold


def create_predictions_from_model(
    emails: list,
    model_function
) -> dict:
    """
    Create predictions from your model.
    
    Args:
        emails: List of email data
        model_function: Function that takes email dict and returns predictions
        
    Returns:
        Dictionary mapping email_id to predictions
    """
    predictions = {}
    
    for email in emails:
        email_id = email["id"]
        try:
            pred = model_function(email)
            
            # Ensure correct format
            predictions[email_id] = {
                "participants": pred.get("participants", []),
                "time": pred.get("time", []),
                "location": pred.get("location", []),
                "topic": pred.get("topic", [])
            }
        except Exception as e:
            logger.warning(f"Error predicting for email {email_id}: {e}")
            predictions[email_id] = {
                "participants": [],
                "time": [],
                "location": [],
                "topic": []
            }
    
    return predictions


def evaluate_project_models(
    gold_annotations_file: str,
    model_predictions_dir: str,
    output_dir: str,
    model_names: list = None
) -> None:
    """
    Evaluate multiple models from your project.
    
    Args:
        gold_annotations_file: Path to gold.json
        model_predictions_dir: Directory with predictions JSONs
        output_dir: Where to save results
        model_names: List of model names to evaluate
    """
    
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("EVALUATING PROJECT MODELS")
    logger.info("="*80)
    
    # 1. Load gold annotations
    logger.info("\n1. Loading gold annotations...")
    gold = load_project_gold_annotations(gold_annotations_file)
    
    if not gold:
        logger.error("Failed to load gold annotations")
        return
    
    logger.info(f"Loaded {len(gold)} emails")
    
    # 2. Get list of models to evaluate
    if not model_names:
        pred_dir = Path(model_predictions_dir)
        model_files = list(pred_dir.glob("*_predictions.json"))
        model_names = [f.stem.replace("_predictions", "") for f in model_files]
    
    if not model_names:
        logger.warning("No models found to evaluate")
        return
    
    logger.info(f"Found models: {model_names}")
    
    # 3. Evaluate each model
    logger.info("\n2. Evaluating models...")
    
    all_results = {}
    
    for model_name in model_names:
        logger.info(f"\n  Evaluating: {model_name}")
        
        # Load predictions
        pred_file = Path(model_predictions_dir) / f"{model_name}_predictions.json"
        
        if not pred_file.exists():
            logger.warning(f"  ✗ Predictions file not found: {pred_file}")
            continue
        
        try:
            predictions = DataLoader.load_predictions(str(pred_file))
            
            # Validate predictions
            DataValidator.validate_predictions(predictions)
            
            # Merge gold with predictions
            emails_data = DataLoader.merge_gold_and_predictions(gold, predictions)
            
            logger.info(f"  ✓ Merged {len(emails_data)} emails")
            
            # Evaluate
            evaluator = ArgumentExtractionEvaluator(
                normalize_text=True,
                partial_match_threshold=0.7,
                verbose=False
            )
            
            results = evaluator.evaluate_batch(emails_data)
            all_results[model_name] = results
            
            logger.info(f"  ✓ Evaluation complete")
            
            # Save model-specific results
            model_output_dir = output_path / model_name
            model_output_dir.mkdir(exist_ok=True)
            
            evaluator.save_results(str(model_output_dir / "results.json"))
            evaluator.save_errors(str(model_output_dir / "errors.json"))
            
        except Exception as e:
            logger.error(f"  ✗ Error evaluating {model_name}: {e}")
            continue
    
    if not all_results:
        logger.error("No models evaluated successfully")
        return
    
    # 4. Generate reports
    logger.info("\n3. Generating reports...")
    
    report_gen = ReportGenerator(verbose=True)
    
    for model_name, results in all_results.items():
        model_output_dir = output_path / model_name
        
        report_gen.generate_json_report(
            results,
            str(model_output_dir / "report.json")
        )
        
        report_gen.generate_markdown_report(
            results,
            str(model_output_dir / "report.md"),
            model_name=model_name
        )
        
        report_gen.generate_csv_report(
            results,
            str(model_output_dir / "metrics.csv"),
            report_type="metrics"
        )
        
        report_gen.generate_latex_table(
            results,
            str(model_output_dir / "table.tex")
        )
    
    # 5. Comparative reports
    logger.info("\n4. Generating comparative reports...")
    
    comparison_dir = output_path / "comparison"
    comparison_dir.mkdir(exist_ok=True)
    
    report_gen.generate_comparative_report(
        all_results,
        str(comparison_dir / "comparison.md"),
        output_format="markdown"
    )
    
    report_gen.generate_comparative_report(
        all_results,
        str(comparison_dir / "comparison.csv"),
        output_format="csv"
    )
    
    # 6. Create visualizations
    logger.info("\n5. Generating visualizations...")
    
    try:
        visualizer = Visualizer()
        
        for model_name, results in all_results.items():
            model_output_dir = output_path / model_name / "visualizations"
            model_output_dir.mkdir(parents=True, exist_ok=True)
            
            visualizer.generate_all_visualizations(
                results,
                str(model_output_dir)
            )
        
        # Comparative visualizations
        comp_viz_dir = comparison_dir / "visualizations"
        comp_viz_dir.mkdir(parents=True, exist_ok=True)
        
        visualizer.plot_model_comparison(
            all_results,
            str(comp_viz_dir / "f1_comparison.png"),
            metric="f1"
        )
        
        visualizer.plot_f1_heatmap(
            all_results,
            str(comp_viz_dir / "f1_heatmap.png")
        )
        
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")
    
    # 7. Print summary
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    
    # Rankings
    logger.info("\nModel Rankings (by Micro F1):\n")
    
    rankings = []
    for model_name, results in all_results.items():
        report = results.get("evaluation_report", {})
        micro_f1 = report.get("aggregated_metrics", {}).get("micro_f1", 0)
        rankings.append((model_name, micro_f1))
    
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model_name, score) in enumerate(rankings, 1):
        print(f"{i}. {model_name:20s} → F1: {score:.4f}")
    
    print()
    
    # Per-argument performance
    logger.info("Per-Argument Performance (Micro F1):\n")
    
    arg_scores = {
        "participants": [],
        "time": [],
        "location": [],
        "topic": []
    }
    
    for model_name, results in all_results.items():
        report = results.get("evaluation_report", {})
        per_class = report.get("per_class_metrics", {})
        
        for arg_type in arg_scores.keys():
            arg_scores[arg_type].append(
                (model_name, per_class.get(arg_type, {}).get("f1", 0))
            )
    
    for arg_type, scores in arg_scores.items():
        scores.sort(key=lambda x: x[1], reverse=True)
        print(f"{arg_type}:")
        for model_name, score in scores:
            print(f"  {model_name:20s} → {score:.4f}")
        print()
    
    logger.info("="*80 + "\n")
    logger.info(f"✓ All results saved to: {output_path.absolute()}\n")


def main():
    """
    Main entry point for project evaluation.
    
    Customize these paths to match your project structure.
    """
    
    # Configure paths
    gold_file = "gold_annotations/output/gold.json"
    predictions_dir = "model_predictions"  # Directory with *_predictions.json
    output_dir = "evaluation_results"
    
    # Model names to evaluate (optional - auto-detect if not specified)
    model_names = [
        "regex",
        "spacy",
        "qa",
        "transformer"
    ]
    
    # Run evaluation
    evaluate_project_models(
        gold_annotations_file=gold_file,
        model_predictions_dir=predictions_dir,
        output_dir=output_dir,
        model_names=model_names
    )


if __name__ == "__main__":
    main()
