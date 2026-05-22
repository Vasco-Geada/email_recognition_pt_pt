# Decision Tree Baseline

This module adds a Decision Tree baseline for PT-PT email intent classification.

Default dataset:

```bash
dataset/realistic_emails_v2.json
```

Files:

- `models/decision_tree_classifier.py`
- `models/train_decision_tree.py`
- `models/predict_decision_tree.py`
- `models/evaluate_decision_tree.py`

## Train

```bash
python models/train_decision_tree.py
```

This saves:

- `models/decision_tree_model.joblib`
- `models/decision_tree_vectorizer.joblib`

## Predict

```bash
python models/predict_decision_tree.py --text "Boas Ana, podemos reunir amanha as 15h?"
```

Example output:

```json
{
  "prediction": "agendamento_reuniao",
  "probabilities": {
    "agendamento_reuniao": 1.0,
    "cancelamento_reuniao": 0.0,
    "reuniao_confirmada": 0.0
  },
  "confidence": 1.0
}
```

## Evaluate

```bash
python models/evaluate_decision_tree.py
```

Recommended comparison metrics:

- accuracy
- macro F1
- weighted F1

## Dissertation Notes

Advantages:

- Highly interpretable.
- Can expose globally important TF-IDF features.
- Fast to train and easy to reproduce.
- Useful as a classical baseline.

Limitations:

- Sparse TF-IDF features create high-dimensional input, which trees handle poorly compared with linear models.
- Decision boundaries are axis-aligned, so the model may overfit specific words or n-grams.
- Small wording changes in informal emails can cause unstable splits.
- Probabilities are often poorly calibrated.

Why it may perform worse than Logistic Regression:

- Logistic Regression uses all features jointly through weighted linear combinations.
- Decision Trees split greedily and may not generalize well in sparse text spaces.
- TF-IDF text classification often benefits from linear separators, especially with short documents.

How to use in the dissertation:

- Present it as an interpretable baseline, not as the expected best model.
- Compare against Naive Bayes, Logistic Regression and BERTimbau using the same split and same dataset.
- Report accuracy, macro F1 and weighted F1.
- Discuss whether errors concentrate in `cancelamento_reuniao` vs `reuniao_confirmada`, since both can contain availability/confirmation language.

