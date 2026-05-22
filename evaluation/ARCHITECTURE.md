# Argument Extraction Evaluation Framework - Architecture

## Overview

The evaluation framework is built on a modular, layered architecture designed for:
- **Flexibility**: Support multiple models and evaluation strategies
- **Extensibility**: Easy to add new metrics or span matching algorithms
- **Clarity**: Each component has a single responsibility
- **Reproducibility**: Full configuration tracking and deterministic evaluation
- **Academic rigor**: Following standard NLP evaluation protocols

## Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│                   User Interface Layer                   │
│  (ModelComparator, ReportGenerator, Visualizer)          │
├─────────────────────────────────────────────────────────┤
│                  Orchestration Layer                     │
│  (ArgumentExtractionEvaluator)                          │
├─────────────────────────────────────────────────────────┤
│                    Computation Layer                     │
│  (SpanMatcher, MetricsCalculator)                       │
├─────────────────────────────────────────────────────────┤
│                    Data Layer                            │
│  (DataLoader, DataValidator, DataPreprocessor)          │
└─────────────────────────────────────────────────────────┘
```

## Component Descriptions

### 1. Data Layer (utils.py)

**Purpose**: Data loading, validation, and preprocessing

**Key Classes**:
- `DataLoader`: Load gold annotations and predictions from JSON
- `DataValidator`: Validate data structure and content
- `DataPreprocessor`: Clean and normalize argument text

**Responsibilities**:
- Read JSON files
- Check data integrity
- Normalize whitespace and encoding
- Convert between formats

**Dependencies**: None (only stdlib)

### 2. Computation Layer

#### 2a. span_matching.py - Span Matching

**Purpose**: Compare individual argument spans

**Key Classes**:
- `TextNormalizer`: Normalize text for comparison
- `TokenOverlapMatcher`: Token-level overlap calculation
- `CharacterOverlapMatcher`: Character-level Jaccard similarity
- `SpanMatcher`: Orchestrate span matching

**Algorithms**:
1. **Text Normalization**: Lowercase, Unicode normalization, whitespace
2. **Token Overlap**: Count intersecting tokens
3. **Jaccard Similarity**: Set-based similarity
4. **List Matching**: Greedy best-match algorithm for argument lists

**Data Flow**:
```
Input: gold_text, predicted_text
  ↓
Normalize both texts
  ↓
Check exact match
  ↓
If not exact:
  - Calculate token overlap
  - Calculate character-level metrics
  - Determine match type (exact/partial/fuzzy/none)
  ↓
Output: SpanMatch with detailed metrics
```

#### 2b. metrics.py - Metrics Calculation

**Purpose**: Calculate evaluation metrics

**Key Classes**:
- `ConfusionMetrics`: Store TP/FP/FN/TN
- `ClassMetrics`: Per-argument-type metrics
- `AggregatedMetrics`: Overall metrics
- `MetricsCalculator`: Calculate precision, recall, F1
- `MatchAwareMetricsCalculator`: Handle different match types

**Metrics**:
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1: Harmonic mean
- Accuracy: Total correct / total
- Micro/Macro/Weighted averages

**Aggregation Methods**:
- **MICRO**: Pool all TP/FP/FN → compute once
- **MACRO**: Per-class metrics → average
- **WEIGHTED**: Macro average weighted by support

### 3. Orchestration Layer (evaluate_arguments.py)

**Purpose**: Coordinate evaluation workflow

**Key Classes**:
- `EmailEvaluationResult`: Result for single email
- `ErrorInstance`: Single error for analysis
- `ArgumentExtractionEvaluator`: Main orchestrator

**Workflow**:
```
1. Load email and predictions
2. For each argument type:
   a. Match gold vs predicted spans (using SpanMatcher)
   b. Count TP, FP, FN
   c. Calculate metrics (using MetricsCalculator)
   d. Record errors (using ErrorInstance)
3. Aggregate results
4. Perform error analysis
```

**Error Categorization**:
- **False Negatives**: Gold spans not predicted
- **False Positives**: Predicted spans not in gold
- **Partial Matches**: Matches with Jaccard < 1.0

### 4. User Interface Layer

#### 4a. report_generator.py - Report Generation

**Purpose**: Generate reports in multiple formats

**Output Formats**:
- **JSON**: Complete structured data
- **CSV**: Tabular format (spreadsheets)
- **Markdown**: Human-readable documentation
- **LaTeX**: Academic publication tables

**Report Types**:
- Per-model: Individual model reports
- Comparative: Multiple model comparison
- Summary: Statistics and rankings

#### 4b. visualization.py - Visualizations

**Purpose**: Generate plots and charts

**Plot Types**:
1. **Bar Charts**: Per-argument metrics, error counts
2. **Confusion Matrices**: TP/FP/FN/TN visualization
3. **Heatmaps**: Models vs argument types
4. **Pie Charts**: Error distribution
5. **Comparison Plots**: Model rankings

**Plotting Library**: matplotlib (+ optional seaborn)

#### 4c. model_comparison.py - Model Comparison

**Purpose**: Compare multiple models

**Features**:
- Register multiple models
- Evaluate all models
- Rank by different metrics
- Identify strengths/weaknesses
- Generate comparative reports

**Ranking Metrics**:
- Micro F1
- Macro F1
- Weighted F1
- Argument-specific F1

## Data Flow - Complete Workflow

```
1. Input Data
   ├── gold_annotations.json
   │   └── DataLoader.load_gold_annotations()
   ├── predictions.json (per model)
   │   └── DataLoader.load_predictions()
   └── merged data
       └── DataLoader.merge_gold_and_predictions()

2. Preprocessing
   └── DataPreprocessor.clean_email_data()
       ├── Trim whitespace
       ├── Normalize Unicode
       └── Remove duplicates

3. Evaluation (per email)
   ├── ArgumentExtractionEvaluator.evaluate_single_email()
   ├── For each argument type:
   │   ├── SpanMatcher.match_lists()
   │   │   ├── TextNormalizer.normalize()
   │   │   ├── TokenOverlapMatcher.token_overlap()
   │   │   └── CharacterOverlapMatcher.jaccard_similarity()
   │   ├── MetricsCalculator.calculate_class_metrics()
   │   └── Record errors
   └── Aggregate results

4. Aggregation
   └── MetricsCalculator.aggregate_metrics()
       ├── Micro average (pool)
       ├── Macro average (mean)
       └── Weighted F1

5. Error Analysis
   └── _summarize_errors()
       ├── Group by error type
       ├── Group by argument type
       └── Identify patterns

6. Output Generation
   ├── ReportGenerator
   │   ├── JSON report
   │   ├── CSV tables
   │   ├── Markdown report
   │   └── LaTeX table
   ├── Visualizer
   │   ├── Metrics plots
   │   ├── Error plots
   │   └── Confusion matrices
   └── ModelComparator
       ├── Rankings
       ├── Comparative reports
       └── Heatmaps
```

## Design Patterns

### 1. Factory Pattern
- `MetricsCalculator.calculate_class_metrics()` creates ClassMetrics
- `SpanMatcher.match()` creates SpanMatch

### 2. Strategy Pattern
- Multiple span matching strategies (exact, partial, fuzzy)
- Multiple aggregation methods (micro, macro, weighted)
- Multiple output formats (JSON, CSV, Markdown, LaTeX)

### 3. Builder Pattern
- `ArgumentExtractionEvaluator` builds evaluation results step-by-step
- `ModelComparator` accumulates models then evaluates

### 4. Composite Pattern
- `AggregatedMetrics` combines `ClassMetrics`
- `EvaluationReport` combines multiple result types

### 5. Adapter Pattern
- Data adapters normalize different input formats
- Output formatters adapt results to target formats

## Key Design Decisions

### 1. Why separate span matching from metrics?
- **Reusability**: Span matching can be used independently
- **Testability**: Each component testable in isolation
- **Flexibility**: Easy to swap matching algorithms

### 2. Why multiple match types (exact, partial, fuzzy)?
- **Fairness**: Different perspectives on correctness
- **Analysis**: Understand what "almost correct" means
- **Tunability**: Trade-off between strictness and leniency

### 3. Why per-argument-type metrics?
- **Interpretability**: Identify argument-specific challenges
- **Comparison**: See which arguments are harder
- **Improvement**: Target weak areas

### 4. Why macro and micro averages?
- **Macro**: Fair if all argument types equally important
- **Micro**: Fair if some types more common (weighted by frequency)
- **Both**: Comprehensive view of performance

### 5. Why comprehensive error analysis?
- **Debugging**: Understand where models fail
- **Improvement**: Guide model development
- **Documentation**: Record issues for dissertation

## Extensibility Points

### Add new metrics
1. Add calculation method to `MetricsCalculator`
2. Add to `ClassMetrics` or `AggregatedMetrics`
3. Add to report generator output

### Add new span matching
1. Create new matcher class (inherit from base)
2. Implement `match()` method
3. Use in `SpanMatcher`

### Add new output format
1. Add method to `ReportGenerator`
2. Follow existing format conventions
3. Test with sample data

### Add new visualization
1. Add method to `Visualizer`
2. Use matplotlib for plotting
3. Save to output directory

## Performance Considerations

### Time Complexity
- Single email evaluation: O(n×m) where n=gold spans, m=predictions
- Batch evaluation: O(emails × argument_types × n×m)
- Model comparison: O(models × batch_evaluation)

### Memory Complexity
- Stores all results in memory
- For large datasets, consider batch processing

### Optimization Tips
1. Process in batches if dataset is large
2. Reuse same evaluator for multiple batches
3. Generate reports/visualizations separately

## Testing Strategy

### Unit Tests
- Test each matcher type with known inputs
- Test metric calculations with known results
- Test data validation with various formats

### Integration Tests
- End-to-end single model evaluation
- Multiple model comparison
- Report generation

### Regression Tests
- Evaluate golden dataset
- Compare results across versions

## Documentation Structure

1. **README.md**: Overview and quick start
2. **QUICKSTART.md**: Step-by-step examples
3. **ARCHITECTURE.md** (this file): Design and components
4. **Inline docstrings**: Implementation details
5. **example_usage.py**: Complete working example

## References

### Evaluation Standards
- CoNLL evaluation guidelines
- SemEval shared task protocols
- ACL recommendation on metrics

### Academic Papers
- Information Extraction evaluation (Nadeau & Sekine, 2007)
- Named Entity Recognition metrics (Tjong & De Meulder, 2003)
- Error analysis in NLP (Rehm et al., 2001)

## Version History

### v1.0.0 (May 2026)
- Initial release
- Complete evaluation framework
- Multiple output formats
- Model comparison
- Visualizations
- Portuguese language support
