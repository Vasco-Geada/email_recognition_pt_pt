"""
QUICKSTART - Argument Extraction Evaluation Framework

Get started in 5 minutes.

## Scenario 1: Evaluate a Single Model

### Step 1: Prepare your data

gold_annotations.json:
```json
[
  {
    "id": 1,
    "text": "Email text here",
    "intent": "agendamento_reuniao",
    "arguments": {
      "participants": ["Ana"],
      "time": ["amanhã às 15h"],
      "location": ["Teams"],
      "topic": []
    }
  }
]
```

predictions.json:
```json
{
  "1": {
    "participants": ["Ana"],
    "time": ["15h"],
    "location": ["Teams"],
    "topic": []
  }
}
```

### Step 2: Run evaluation

```python
from evaluation import ArgumentExtractionEvaluator, DataLoader, ReportGenerator, Visualizer

# Load data
gold = DataLoader.load_gold_annotations("gold_annotations.json")
predictions = DataLoader.load_predictions("predictions.json")
merged = DataLoader.merge_gold_and_predictions(gold, predictions)

# Evaluate
evaluator = ArgumentExtractionEvaluator()
results = evaluator.evaluate_batch(merged)

# Generate reports
report_gen = ReportGenerator()
report_gen.generate_markdown_report(results, "report.md", model_name="My Model")
report_gen.generate_csv_report(results, "metrics.csv")

# Visualize
visualizer = Visualizer()
visualizer.generate_all_visualizations(results, "visualizations/")

print("✅ Evaluation complete!")
```

---

## Scenario 2: Compare Multiple Models

```python
from evaluation import ModelComparator, ArgumentExtractionEvaluator, DataLoader

# Load gold annotations
gold = DataLoader.load_gold_annotations("gold_annotations.json")

# Load predictions for each model
model1_preds = DataLoader.load_predictions("model1_predictions.json")
model2_preds = DataLoader.load_predictions("model2_predictions.json")
model3_preds = DataLoader.load_predictions("model3_predictions.json")

# Prepare data
models_data = {
    "Model1": DataLoader.merge_gold_and_predictions(gold, model1_preds),
    "Model2": DataLoader.merge_gold_and_predictions(gold, model2_preds),
    "Model3": DataLoader.merge_gold_and_predictions(gold, model3_preds),
}

# Create comparator
comparator = ModelComparator()

# Evaluate each model
for model_name, data in models_data.items():
    evaluator = ArgumentExtractionEvaluator()
    comparator.models_results[model_name] = evaluator.evaluate_batch(data)

# Print rankings
print("\n=== Model Rankings ===\n")
rankings = comparator.get_model_rankings("micro_f1")
for i, (model, score) in enumerate(rankings, 1):
    print(f"{i}. {model}: {score:.4f}")

# Print summary
comparator.print_comparison_summary()

# Save results
comparator.save_comparison_results("comparison_results.json")

print("✅ Comparison complete!")
```

---

## Scenario 3: Error Analysis

```python
from evaluation import ArgumentExtractionEvaluator, DataLoader

# Load and evaluate
gold = DataLoader.load_gold_annotations("gold_annotations.json")
predictions = DataLoader.load_predictions("predictions.json")
merged = DataLoader.merge_gold_and_predictions(gold, predictions)

evaluator = ArgumentExtractionEvaluator()
results = evaluator.evaluate_batch(merged)

# Analyze errors
print("\n=== Error Analysis ===\n")

# 1. False Negatives (missed spans)
print("False Negatives (missed predictions):")
false_negatives = [e for e in evaluator.errors if e.error_type == "false_negative"]
for error in false_negatives[:5]:
    print(f"  - {error.argument_type}: '{error.gold_value}'")

# 2. False Positives (incorrect predictions)
print("\nFalse Positives (incorrect predictions):")
false_positives = [e for e in evaluator.errors if e.error_type == "false_positive"]
for error in false_positives[:5]:
    print(f"  - {error.argument_type}: '{error.predicted_value}'")

# 3. Partial Matches (partial correctness)
print("\nPartial Matches (partial correctness):")
partial_matches = [e for e in evaluator.errors if e.error_type == "partial_match"]
for error in partial_matches[:5]:
    print(f"  - {error.argument_type}:")
    print(f"      Gold: '{error.gold_value}'")
    print(f"      Pred: '{error.predicted_value}'")
    print(f"      Jaccard: {error.context.get('jaccard_similarity', 0):.3f}")

# Save error analysis
evaluator.save_errors("error_analysis.json")

print("\n✅ Error analysis complete!")
```

---

## Output Files

After running evaluation, you'll have:

### JSON Files
- `results.json` - Complete evaluation data
- `errors.json` - Detailed error analysis
- `comparison_results.json` - Multi-model comparison

### CSV Files
- `metrics.csv` - Tabular metrics
- `errors.csv` - Errors in table format
- `emails.csv` - Per-email scores

### Reports
- `report.md` - Markdown report (for reading)
- `table.tex` - LaTeX table (for dissertation)

### Visualizations
- `metrics_f1.png` - F1 scores by argument
- `precision_recall_f1.png` - All metrics
- `confusion_matrix_*.png` - Per-argument confusion matrices
- `error_distribution.png` - Error type distribution
- `error_by_argument.png` - Errors by argument type
- `f1_comparison.png` - Model comparison
- `f1_heatmap.png` - Models × Arguments matrix

---

## Interpretation Guide

### Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Precision** | TP/(TP+FP) | Of predicted arguments, how many are correct? |
| **Recall** | TP/(TP+FN) | Of gold arguments, how many were found? |
| **F1** | 2×(P×R)/(P+R) | Balanced measure of precision and recall |
| **Micro F1** | Pooled metrics | Overall performance |
| **Macro F1** | Average per-argument | Per-argument average |

### What's Good?

- **Micro F1 > 0.85**: Good model performance
- **Macro F1 > 0.80**: Consistent across argument types
- **Balanced P and R**: No overfitting to specific arguments
- **Low variance between models**: Stable approach

### What to Fix?

- **Low Recall**: Model missing many arguments → improve detection
- **Low Precision**: Model predicting too many incorrect arguments → improve accuracy
- **Argument-specific issues**: Some arguments (e.g., "time") much harder → special handling?
- **High false negatives**: Model too conservative
- **High false positives**: Model too aggressive

---

## Common Issues

### Issue: Import errors

```
ModuleNotFoundError: No module named 'evaluation'
```

**Solution**: Make sure you're in the project root directory and the evaluation/ folder exists.

### Issue: Data format errors

```
ValueError: Missing required columns
```

**Solution**: Check that your JSON files have the exact structure shown in the examples.

### Issue: Visualization errors

```
ModuleNotFoundError: No module named 'matplotlib'
```

**Solution**: Install with: `pip install -r evaluation/requirements.txt`

### Issue: Low scores

**Solution**: Check:
1. Is your gold data correct?
2. Are predictions in the right format?
3. Try different thresholds: `ArgumentExtractionEvaluator(partial_match_threshold=0.5)`

---

## Next Steps

1. **Read the full README.md** for advanced features
2. **Check example_usage.py** for complete workflow
3. **Customize thresholds** for your use case
4. **Analyze errors** to improve your model
5. **Generate dissertation figures** with visualization.py

---

## Need Help?

1. Check example_usage.py for complete working example
2. Read docstrings in each module
3. Look at README.md in evaluation/ folder
4. Validate data with DataValidator.validate_*()
"""
