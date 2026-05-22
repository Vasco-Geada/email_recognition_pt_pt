"""
Argument Extraction Evaluation Framework

Comprehensive scientific evaluation system for comparing argument extraction 
models in Portuguese academic emails related to meeting scheduling.

## 🎯 Overview

This framework provides a complete evaluation pipeline for argument extraction 
models, supporting:

- **Multi-argument-type evaluation**: participants, time, location, topic
- **Multiple span matching strategies**: exact, partial, token overlap, fuzzy
- **Comprehensive metrics**: Precision, Recall, F1, Micro/Macro/Weighted averages
- **Per-argument analysis**: Individual metrics for each argument type
- **Error analysis**: Automatic categorization of false positives, negatives, partial matches
- **Model comparison**: Side-by-side comparison of multiple models
- **Multiple output formats**: JSON, CSV, Markdown, LaTeX
- **Visualizations**: Bar charts, confusion matrices, heatmaps, error distributions

## 📁 Structure

```
evaluation/
├── span_matching.py          # Span matching implementations
├── metrics.py               # Metrics calculation
├── evaluate_arguments.py    # Main evaluator
├── report_generator.py      # Report generation (JSON, CSV, MD, LaTeX)
├── visualization.py         # Plotting and visualizations
├── model_comparison.py      # Multi-model comparison
├── utils.py                 # Utility functions
├── __init__.py              # Package initialization
├── example_usage.py         # Complete usage example
└── README.md                # This file
```

## 🚀 Quick Start

### 1. Single Model Evaluation

```python
from evaluation import ArgumentExtractionEvaluator

# Create evaluator
evaluator = ArgumentExtractionEvaluator(
    normalize_text=True,
    partial_match_threshold=0.7,
    verbose=True
)

# Prepare data (gold annotations + predictions)
emails_data = [
    {
        "id": 1,
        "text": "Email text here",
        "intent": "agendamento_reuniao",
        "arguments": {
            "participants": ["Ana"],
            "time": ["amanhã às 15h"],
            "location": ["Teams"],
            "topic": []
        },
        "predicted": {
            "participants": ["Ana"],
            "time": ["15h"],
            "location": ["Teams"],
            "topic": []
        }
    }
]

# Evaluate
results = evaluator.evaluate_batch(emails_data)

# Save
evaluator.save_results("results.json")
evaluator.save_errors("errors.json")
```

### 2. Generate Reports

```python
from evaluation import ReportGenerator

report_gen = ReportGenerator()

# Generate various formats
report_gen.generate_json_report(results, "report.json")
report_gen.generate_csv_report(results, "metrics.csv")
report_gen.generate_markdown_report(results, "report.md")
report_gen.generate_latex_table(results, "table.tex")
```

### 3. Create Visualizations

```python
from evaluation import Visualizer

visualizer = Visualizer()

# Generate all visualizations
visualizer.generate_all_visualizations(results, "output_dir")

# Or specific plots
visualizer.plot_per_argument_metrics(results, "f1_scores.png")
visualizer.plot_confusion_matrix(results, "participants", "cm.png")
```

### 4. Compare Multiple Models

```python
from evaluation import ModelComparator

comparator = ModelComparator()

# Register models
comparator.register_model("model1", evaluator1)
comparator.register_model("model2", evaluator2)
comparator.register_model("model3", evaluator3)

# Evaluate all
results = comparator.evaluate_all_models(emails_data)

# Get rankings
rankings = comparator.get_model_rankings("micro_f1")
for model_name, score in rankings:
    print(f"{model_name}: {score:.4f}")

# Print summary
comparator.print_comparison_summary()

# Save results
comparator.save_comparison_results("comparison.json")
```

## 📊 Data Format

### Gold Annotations (Required)

```json
[
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
  }
]
```

**Required fields:**
- `id`: Unique email identifier
- `text`: Full email text
- `intent`: Email intent (agendamento_reuniao, cancelamento_reuniao, reuniao_confirmada)
- `arguments`: Dictionary with argument types as keys, lists of spans as values

**Argument types:**
- `participants`: Person names/pronouns
- `time`: Temporal expressions (dates, times, relative references)
- `location`: Physical or virtual meeting locations
- `topic`: Meeting topics/subjects

### Predictions Format

```json
{
  "participants": ["Ana"],
  "time": ["15h"],
  "location": ["Teams"],
  "topic": []
}
```

Models should return dictionary with same structure as `arguments`.

## 📈 Metrics

### Per-Argument-Type Metrics

- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Support**: Total number of gold instances

### Aggregated Metrics

- **Micro Average**: Pool all TP/FP/FN, then compute metrics
- **Macro Average**: Compute per-class, then average
- **Weighted F1**: Weighted by support (number of gold instances)
- **Accuracy**: Total correct / total instances

### Match Types

1. **Exact Match**: Normalized texts are identical
2. **Partial Match**: Jaccard similarity ≥ partial_match_threshold (default: 0.7)
3. **Fuzzy Match**: Jaccard similarity ≥ fuzzy_match_threshold (default: 0.6)
4. **No Match**: Jaccard similarity < fuzzy_match_threshold

## 🔧 Configuration

### Evaluator Options

```python
evaluator = ArgumentExtractionEvaluator(
    # Thresholds for different match types
    exact_match_threshold=1.0,          # Exact only
    partial_match_threshold=0.7,        # Partial
    fuzzy_match_threshold=0.6,          # Fuzzy
    
    # Text normalization
    normalize_text=True,                # Lowercase + whitespace normalization
    remove_accents=False,               # Keep Portuguese accents (ã, ç, é)
    remove_punctuation=False,           # Keep punctuation
    
    # Verbosity
    verbose=True
)
```

### Visualization Options

```python
visualizer = Visualizer(
    figsize=(12, 8),          # Figure size
    dpi=300,                  # Resolution
    style="seaborn",          # matplotlib style
    font_size=10
)
```

## 📋 Output Files

### Per-Model Outputs

- **results.json**: Complete evaluation results in JSON
- **errors.json**: Detailed error analysis
- **report.json**: Structured report
- **report.md**: Human-readable Markdown report
- **metrics.csv**: Metrics in tabular format
- **errors.csv**: Errors in tabular format
- **emails.csv**: Per-email metrics
- **table.tex**: LaTeX table for dissertation
- **visualizations/**: Directory with PNG plots

### Comparative Outputs

- **comparison_results.json**: Comparison data
- **comparison.md**: Comparative Markdown report
- **comparison.csv**: Comparative CSV report
- **visualizations/**: Comparative plots

## 🎨 Visualization Types

1. **Per-argument metrics**: Bar chart of F1 scores
2. **Precision-Recall-F1**: Grouped bar chart
3. **Confusion matrices**: Per-argument confusion matrix
4. **Error distribution**: Pie and bar charts
5. **Error by argument**: Error count per argument type
6. **Model comparison**: F1 scores across models
7. **F1 heatmap**: Models vs argument types matrix

## 📚 Utility Functions

### Data Loading

```python
from evaluation import DataLoader

# Load gold annotations
gold = DataLoader.load_gold_annotations("gold.json")

# Load predictions
preds = DataLoader.load_predictions("predictions.json")

# Merge
merged = DataLoader.merge_gold_and_predictions(gold, preds)
```

### Data Validation

```python
from evaluation import DataValidator

DataValidator.validate_gold_annotations(gold)
DataValidator.validate_predictions(preds)
```

### Data Preprocessing

```python
from evaluation import DataPreprocessor

cleaned = DataPreprocessor.clean_email_data(email_data)
```

## 🎯 Advanced Usage

### Custom Thresholds

```python
# Stricter evaluation (only exact matches count)
strict_evaluator = ArgumentExtractionEvaluator(
    partial_match_threshold=1.0,
    fuzzy_match_threshold=1.0
)

# Lenient evaluation (more partial matches accepted)
lenient_evaluator = ArgumentExtractionEvaluator(
    partial_match_threshold=0.5,
    fuzzy_match_threshold=0.3
)
```

### Error Analysis

```python
# Get all false negatives
false_negatives = [e for e in evaluator.errors 
                   if e.error_type == "false_negative"]

# Get errors for specific argument type
time_errors = [e for e in evaluator.errors 
               if e.argument_type == "time"]

# Get partial matches
partial_matches = [e for e in evaluator.errors 
                   if e.error_type == "partial_match"]
```

### Model Strengths/Weaknesses

```python
comparator = ModelComparator()
# ... evaluate models ...

for model_name in comparator.models_results.keys():
    analysis = comparator.identify_model_strengths_weaknesses(model_name)
    print(f"\n{model_name}:")
    print(f"  Best: {analysis['best_argument_type']}")
    print(f"  Worst: {analysis['worst_argument_type']}")
```

## 📝 Example Report Output

### Markdown Report

```markdown
# Evaluation Report: Regex Model

**Generated**: 2026-05-22 10:30:45

## Summary Statistics

- Total emails evaluated: 100
- Total errors: 25
- False positives: 8
- False negatives: 12
- Partial matches: 5

## Per-Argument-Type Metrics

| Argument Type | Precision | Recall | F1 | Support |
|---------------|-----------|--------|-----|---------|
| participants  | 0.9200    | 0.8800 | 0.8996 | 50 |
| time          | 0.8500    | 0.9000 | 0.8744 | 60 |
| location      | 0.7800    | 0.7500 | 0.7647 | 40 |
| topic         | 0.6500    | 0.5500 | 0.5962 | 20 |
```

### LaTeX Table

```latex
\begin{table}[h]
\centering
\begin{tabular}{|l|r|r|r|r|}
\hline
\textbf{Argument Type} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{Support} \\
\hline
participants & 0.9200 & 0.8800 & 0.8996 & 50 \\
time & 0.8500 & 0.9000 & 0.8744 & 60 \\
location & 0.7800 & 0.7500 & 0.7647 & 40 \\
topic & 0.6500 & 0.5500 & 0.5962 & 20 \\
\hline
\textbf{Micro Avg} & 0.8325 & 0.8275 & 0.8300 & - \\
\textbf{Macro Avg} & 0.8000 & 0.7700 & 0.7837 & - \\
\hline
\end{tabular}
\caption{Argument Extraction Evaluation Metrics}
\label{tab:arg_extraction_metrics}
\end{table}
```

## 🔍 Portuguese Language Support

The framework includes robust support for Portuguese European text:

- **UTF-8 encoding**: Full Unicode support
- **Accent preservation**: Keeps diacritics (ã, ç, é, õ, etc.) by default
- **Informal expressions**: Handles common temporal expressions:
  - "amanhã" (tomorrow)
  - "mais logo" (later)
  - "depois de almoço" (after lunch)
  - "sexta às 15h" (Friday at 3pm)
  - "próxima segunda" (next Monday)

## 📦 Dependencies

```
matplotlib >= 3.5.0
numpy >= 1.21.0
seaborn >= 0.11.0 (optional, for enhanced visualizations)
```

## 🔐 Academic Standards

This framework follows academic evaluation standards:

- CoNLL evaluation guidelines for NER/IE tasks
- SemEval span evaluation protocols
- Confusion matrix analysis
- Macro and micro averaging
- Per-class metrics reporting
- Error analysis for interpretation

## 📖 Citation

If you use this framework in your research, please cite:

```bibtex
@software{arg_extraction_eval_2026,
  title={Argument Extraction Evaluation Framework},
  author={Generated for Email Recognition PT-PT Project},
  year={2026},
  url={https://github.com/...}
}
```

## 📞 Support

For issues or questions:
1. Check the example_usage.py for complete workflow
2. Review inline documentation in each module
3. Check data format examples above
4. Validate with DataValidator

## 📄 License

MIT License - See LICENSE file for details

---

**Framework Version**: 1.0.0  
**Last Updated**: May 2026  
**Status**: Production Ready
"""
