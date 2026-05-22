# Argument Extraction Evaluation Framework - Complete Index

## 📚 Documentation Files

### Main Documentation
1. **README.md** - Framework overview and features
   - What it does
   - Quick start
   - Data formats
   - Configuration options
   - Output files
   - Advanced usage

2. **QUICKSTART.md** - Get started in 5 minutes
   - 3 practical scenarios
   - Copy-paste examples
   - Interpretation guide
   - Common issues

3. **ARCHITECTURE.md** - Design and components
   - Architecture layers
   - Component descriptions
   - Data flow
   - Design patterns
   - Extensibility points

4. **INDEX.md** - This file
   - Navigation guide
   - File descriptions
   - Module organization

## 🗂️ Module Files

### Core Modules

#### 1. span_matching.py
**Purpose**: Implement span matching algorithms

**Key Classes**:
- `MatchType`: Enum for match types
- `SpanMatch`: Result of span comparison
- `TextNormalizer`: Normalize text
- `TokenOverlapMatcher`: Token-level matching
- `CharacterOverlapMatcher`: Character-level matching
- `SpanMatcher`: Main orchestrator

**Use When**: 
- Implementing new span matching strategies
- Comparing individual spans
- Understanding matching logic

**Lines**: ~500

---

#### 2. metrics.py
**Purpose**: Calculate evaluation metrics

**Key Classes**:
- `ConfusionMetrics`: TP/FP/FN/TN storage
- `ClassMetrics`: Per-class metrics
- `AggregatedMetrics`: Overall metrics
- `AggregationMethod`: Enum for aggregation types
- `MetricsCalculator`: Calculate metrics
- `MatchAwareMetricsCalculator`: Handle match types
- `EvaluationReport`: Complete report structure

**Use When**:
- Computing precision/recall/F1
- Aggregating results
- Building custom metrics

**Lines**: ~400

---

#### 3. evaluate_arguments.py
**Purpose**: Main evaluation orchestrator

**Key Classes**:
- `EmailEvaluationResult`: Result for one email
- `ErrorInstance`: Single error
- `ArgumentExtractionEvaluator`: Main evaluator

**Key Methods**:
- `evaluate_single_email()`: Evaluate one email
- `evaluate_batch()`: Evaluate multiple emails
- `get_aggregated_results()`: Get overall metrics
- `save_results()`: Save to JSON
- `save_errors()`: Save error analysis

**Use When**:
- Evaluating a single model
- Running batch evaluations
- Starting any evaluation task

**Lines**: ~400

---

#### 4. report_generator.py
**Purpose**: Generate reports in multiple formats

**Key Classes**:
- `ReportGenerator`: Main report generator

**Key Methods**:
- `generate_json_report()`: JSON output
- `generate_csv_report()`: CSV output
- `generate_markdown_report()`: Markdown output
- `generate_latex_table()`: LaTeX output
- `generate_comparative_report()`: Compare models

**Use When**:
- Need to export results
- Creating dissertation tables
- Sharing results in different formats

**Lines**: ~500

---

#### 5. visualization.py
**Purpose**: Generate plots and visualizations

**Key Classes**:
- `Visualizer`: Main visualizer

**Key Methods**:
- `plot_per_argument_metrics()`: Bar chart
- `plot_confusion_matrix()`: Confusion matrix
- `plot_precision_recall_f1()`: Grouped bars
- `plot_model_comparison()`: Compare models
- `plot_error_distribution()`: Error pie/bar
- `plot_error_by_argument()`: Error per-argument
- `plot_f1_heatmap()`: Model × argument matrix
- `generate_all_visualizations()`: Generate all plots

**Use When**:
- Creating figures for presentation/dissertation
- Understanding model performance visually
- Debugging model issues

**Lines**: ~600

---

#### 6. model_comparison.py
**Purpose**: Compare multiple models

**Key Classes**:
- `ModelComparator`: Main comparator

**Key Methods**:
- `register_model()`: Add model to comparison
- `evaluate_all_models()`: Evaluate all registered models
- `get_model_rankings()`: Rank models by metric
- `compare_models_on_metric()`: Get scores for all models
- `get_best_model()`: Get top performer
- `compare_on_argument_type()`: Compare per-argument
- `identify_model_strengths_weaknesses()`: Analyze model
- `print_comparison_summary()`: Print to console
- `save_comparison_results()`: Save comparison

**Use When**:
- Comparing 2+ models
- Choosing best model
- Understanding relative performance
- Creating comparison tables

**Lines**: ~400

---

#### 7. utils.py
**Purpose**: Utility functions

**Key Classes**:
- `DataLoader`: Load JSON files
- `DataValidator`: Validate data structure
- `DataPreprocessor`: Clean data
- `EvaluationUtils`: General utilities

**Use When**:
- Loading/validating data
- Cleaning text
- General helper functions

**Lines**: ~400

---

### Supporting Files

#### __init__.py
- Package initialization
- Imports all public classes
- Version information

#### requirements.txt
- Dependencies for evaluation framework
- matplotlib, numpy, pandas, seaborn

#### README.md
- Comprehensive documentation
- Usage examples
- Configuration guide

#### QUICKSTART.md
- 5-minute start guide
- 3 practical scenarios
- Troubleshooting

#### ARCHITECTURE.md
- Design documentation
- Component descriptions
- Data flow diagrams

#### example_usage.py
- Complete working example
- Sample data creation
- Full workflow demonstration

---

## 🔄 Typical Workflows

### Workflow 1: Single Model Evaluation

```
1. Load data (DataLoader)
2. Create evaluator (ArgumentExtractionEvaluator)
3. Evaluate batch (evaluate_batch)
4. Save results (save_results)
5. Generate reports (ReportGenerator)
6. Create visualizations (Visualizer)
```

**Files**: evaluate_arguments.py → report_generator.py → visualization.py

---

### Workflow 2: Multi-Model Comparison

```
1. Load data (DataLoader)
2. Create comparator (ModelComparator)
3. Register models (register_model)
4. Evaluate all (evaluate_all_models)
5. Get rankings (get_model_rankings)
6. Generate comparative reports (report_generator)
7. Create comparative visualizations (visualizer)
```

**Files**: model_comparison.py → report_generator.py → visualization.py

---

### Workflow 3: Error Analysis

```
1. Load data (DataLoader)
2. Evaluate (ArgumentExtractionEvaluator)
3. Access evaluator.errors
4. Categorize by type/argument
5. Export errors (save_errors)
```

**Files**: evaluate_arguments.py → utils.py

---

## 📊 Output Structure

```
evaluation_results/
│
├── model1/
│   ├── results.json          ← All metrics and results
│   ├── errors.json           ← Error analysis
│   ├── report.json           ← Structured report
│   ├── report.md             ← Readable report
│   ├── metrics.csv           ← Tabular metrics
│   ├── errors.csv            ← Error table
│   ├── emails.csv            ← Per-email scores
│   ├── table.tex             ← LaTeX table
│   └── visualizations/
│       ├── metrics_f1.png
│       ├── precision_recall_f1.png
│       ├── confusion_matrix_*.png
│       ├── error_distribution.png
│       └── error_by_argument.png
│
├── model2/
│   ├── [same structure as model1]
│
└── comparison/
    ├── comparison_results.json
    ├── comparison.md
    ├── comparison.csv
    └── visualizations/
        ├── f1_comparison.png
        └── f1_heatmap.png
```

---

## 🎯 Quick Reference

### Load Data
```python
from evaluation import DataLoader
gold = DataLoader.load_gold_annotations("file.json")
preds = DataLoader.load_predictions("file.json")
```

### Evaluate Single Model
```python
from evaluation import ArgumentExtractionEvaluator
evaluator = ArgumentExtractionEvaluator()
results = evaluator.evaluate_batch(data)
```

### Compare Models
```python
from evaluation import ModelComparator
comparator = ModelComparator()
comparator.register_model("m1", eval1)
comparator.evaluate_all_models(data)
comparator.print_comparison_summary()
```

### Generate Reports
```python
from evaluation import ReportGenerator
gen = ReportGenerator()
gen.generate_markdown_report(results, "report.md")
```

### Create Visualizations
```python
from evaluation import Visualizer
viz = Visualizer()
viz.generate_all_visualizations(results, "out/")
```

---

## 🔗 Dependencies Between Modules

```
evaluate_arguments.py
    ├── depends on: span_matching.py
    ├── depends on: metrics.py
    └── produces: EmailEvaluationResult

model_comparison.py
    ├── depends on: evaluate_arguments.py
    ├── depends on: metrics.py
    └── produces: model rankings

report_generator.py
    ├── depends on: evaluate_arguments results
    └── produces: JSON/CSV/MD/LaTeX

visualization.py
    ├── depends on: evaluate_arguments results
    └── produces: PNG plots

utils.py
    ├── no dependencies
    └── used by: all modules

span_matching.py
    ├── no dependencies
    ├── used by: evaluate_arguments.py
    └── used by: metrics.py

metrics.py
    ├── no dependencies
    ├── used by: evaluate_arguments.py
    └── used by: model_comparison.py
```

---

## 📖 Reading Guide

### For First-Time Users
1. Start with **QUICKSTART.md**
2. Run **example_usage.py**
3. Check output files
4. Read **README.md** for details

### For Implementation Details
1. Check specific module docstrings
2. Read **ARCHITECTURE.md**
3. Review key classes
4. Examine example_usage.py

### For Integration
1. Understand data format (README.md)
2. Check evaluate_arguments.py API
3. Learn ModelComparator workflow
4. Test with your data

### For Dissertation
1. Generate reports with report_generator.py
2. Create visualizations with visualization.py
3. Use LaTeX tables directly
4. Reference metrics in ARCHITECTURE.md

---

## 🆘 Finding Help

**I want to...**

| Task | Go to | Module |
|------|-------|--------|
| Evaluate one model | QUICKSTART Scenario 1 | evaluate_arguments.py |
| Compare models | QUICKSTART Scenario 2 | model_comparison.py |
| Analyze errors | QUICKSTART Scenario 3 | evaluate_arguments.py |
| Create reports | README.md | report_generator.py |
| Make plots | README.md | visualization.py |
| Load data | README.md | utils.py |
| Understand metrics | README.md | metrics.py |
| Understand matching | ARCHITECTURE.md | span_matching.py |
| See full example | example_usage.py | (all modules) |
| Troubleshoot | QUICKSTART Common Issues | (varies) |

---

## 📦 File Sizes (Approximate)

| File | Size | Comments |
|------|------|----------|
| span_matching.py | 500 lines | Core matching logic |
| metrics.py | 400 lines | Metric calculations |
| evaluate_arguments.py | 400 lines | Main orchestrator |
| report_generator.py | 500 lines | Multiple formats |
| visualization.py | 600 lines | Plotting code |
| model_comparison.py | 400 lines | Model comparison |
| utils.py | 400 lines | Utility functions |
| example_usage.py | 300 lines | Working example |

**Total**: ~3500 lines of well-documented code

---

## ✅ Checklist for Complete Workflow

- [ ] Install dependencies: `pip install -r evaluation/requirements.txt`
- [ ] Prepare gold annotations JSON
- [ ] Prepare predictions JSON for each model
- [ ] Run example_usage.py to verify setup
- [ ] Evaluate your models
- [ ] Generate reports in needed formats
- [ ] Create visualizations
- [ ] Analyze errors
- [ ] Compare model performance
- [ ] Prepare dissertation figures

---

## 🔗 Cross-References

### Data Format Questions
→ See README.md "Data Format" section

### Metrics Explanation
→ See README.md "Metrics" section and metrics.py docstrings

### Configuration Options
→ See README.md "Configuration" section

### Troubleshooting
→ See QUICKSTART.md "Common Issues" section

### Architecture Details
→ See ARCHITECTURE.md

### Working Examples
→ See example_usage.py

---

**Last Updated**: May 2026  
**Framework Version**: 1.0.0  
**Status**: Production Ready
"""
