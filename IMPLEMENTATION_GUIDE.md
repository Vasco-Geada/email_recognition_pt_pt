"""
IMPLEMENTATION GUIDE: Email Intent Classification Pipeline
===========================================================

This document provides a detailed explanation of each step in the email intent
classification model implementation.

"""

# =============================================================================
# STEP 1: DATA LOADING AND VALIDATION
# =============================================================================

"""
Purpose:
  Load the dataset from a JSON file and validate its structure and content.

Implementation Details:
  - Function: EmailIntentClassifier.load_dataset()
  - Input format: JSON list of objects with 'subject', 'body', 'label' fields
  - Error handling: 
    * FileNotFoundError if dataset doesn't exist
    * JSONDecodeError if JSON is malformed
    * ValueError if required columns are missing

Data Quality Checks:
  - Validates all required columns exist
  - Detects and removes rows with missing values
  - Logs distribution of labels for class imbalance awareness

Code Example:
  >>> classifier = EmailIntentClassifier()
  >>> df = classifier.load_dataset('dataset/dataset.json')
  >>> print(df.shape)  # (n_samples, 4) where columns are subject, body, label, text

Key Features:
  ✓ UTF-8 encoding support for Portuguese characters
  ✓ Automatic handling of missing values
  ✓ Informative logging of data statistics
  ✓ Combines subject and body into single 'text' field for consistency
"""


# =============================================================================
# STEP 2: DATA STRATIFICATION AND SPLITTING
# =============================================================================

"""
Purpose:
  Split data into training (80%) and testing (20%) sets while maintaining
  the class distribution (stratification).

Why Stratification?
  - Ensures both train and test sets have similar class distributions
  - Prevents one set from having all samples of a minority class
  - Results in more reliable evaluation metrics
  - Particularly important with imbalanced datasets

Implementation Details:
  - Function: EmailIntentClassifier.prepare_data()
  - Uses: sklearn.model_selection.train_test_split()
  - Parameters:
    * test_size=0.2 (80% training, 20% testing)
    * stratify=df['label'] (maintains class distribution)
    * random_state=42 (reproducibility)

Code Example:
  >>> train_df, test_df = classifier.prepare_data(df)
  >>> print(f"Train: {len(train_df)}, Test: {len(test_df)}")
  Train: 320, Test: 80
  
  # Both sets will have similar proportions of each intent class
"""


# =============================================================================
# STEP 3: TEXT VECTORIZATION WITH TF-IDF
# =============================================================================

"""
Purpose:
  Convert text data into numerical features that machine learning models can use.

What is TF-IDF?
  TF-IDF = Term Frequency × Inverse Document Frequency
  
  - TF (Term Frequency): How often a word appears in a document
  - IDF (Inverse Document Frequency): How rare a word is across all documents
  - Together: Words that are frequent in one document but rare overall get high scores
  
  Result: Each email text becomes a sparse vector of numerical features

Implementation Details:
  - Function: EmailIntentClassifier.vectorize_text()
  - Uses: sklearn.feature_extraction.text.TfidfVectorizer
  
  Configuration Parameters:
    ✓ lowercase=True
        - Converts all text to lowercase
        - "Meeting" and "meeting" are treated same way
        - Improves consistency
    
    ✓ ngram_range=(1, 2)
        - Unigrams: Single words (e.g., "reunião", "data")
        - Bigrams: Word pairs (e.g., "marcar reunião", "mudar data")
        - Captures both individual words and word combinations
    
    ✓ max_features=5000
        - Limits vocabulary to most important 5000 terms
        - Reduces dimensionality and computational cost
        - Discards very rare words
    
    ✓ strip_accents=None
        - CRITICAL for Portuguese: Preserves accents
        - Keeps: ã, ç, é, ô, etc.
        - Means: "reuniao" and "reunião" are different features
        - Essential for semantic correctness in Portuguese
    
    ✓ min_df=1, max_df=1.0
        - min_df: Include features that appear in at least 1 document
        - max_df: Include features that appear in up to 100% of documents
        - No overly rare or overly common features are removed

Example Transformation:
  Input Text: "Reunião com o cliente amanhã"
  
  Tokenization & Vectorization:
    reunião: 0.45      (appears often, somewhat rare word)
    cliente: 0.35      (moderate frequency)
    com: 0.15          (common word, low weight)
    reunião cliente: 0.30  (bigram)
    cliente amanhã: 0.25   (bigram)
  
  Output: Sparse vector with 5000 dimensions (most values are 0)

Code Example:
  >>> X_train, X_test = classifier.vectorize_text(
  ...     train_texts,
  ...     test_texts,
  ...     max_features=5000,
  ...     ngram_range=(1, 2)
  ... )
  >>> print(X_train.shape)  # (320, 5000)
  >>> print(f"Density: {X_train.nnz / X_train.shape[0] / X_train.shape[1]:.2%}")
  # Shows percentage of non-zero values (~2% typical for text data)
"""


# =============================================================================
# STEP 4: MODEL TRAINING - LOGISTIC REGRESSION
# =============================================================================

"""
Purpose:
  Train a classification model that learns to predict email intent from
  TF-IDF features.

Why Logistic Regression?
  ✓ Fast training and prediction
  ✓ Interpretable (can see feature weights)
  ✓ Works well with high-dimensional sparse text data
  ✓ Probabilistic outputs (confidence scores)
  ✓ Scales well with large feature sets

Implementation Details:
  - Function: EmailIntentClassifier.train_model()
  - Uses: sklearn.linear_model.LogisticRegression
  
  Configuration Parameters:
    ✓ max_iter=1000
        - Maximum iterations for convergence
        - 1000 is sufficient for most text classification tasks
        - Increase if convergence warnings appear
    
    ✓ class_weight='balanced'
        - Automatically adjusts for class imbalance
        - If "agendamento_reuniao" has 200 samples and "nao_reuniao" has 20:
          * Errors on "nao_reuniao" cost 10x more
          * Prevents model from ignoring minority classes
        - Critical for datasets with unbalanced intent classes
    
    ✓ solver='lbfgs'
        - Optimization algorithm for finding best weights
        - Good for small to medium-sized datasets
        - Alternatives: 'liblinear' (faster), 'sag' (online)
    
    ✓ multi_class='multinomial'
        - For multi-class classification (4 intent classes)
        - Computes probabilities over all classes simultaneously
        - Better than one-vs-rest for multinomial problems

What Happens During Training:
  1. Initialize model with random weights W and bias b
  2. For each iteration:
     - Compute predictions for training data: y_hat = sigmoid(X·W + b)
     - Calculate loss (cross-entropy) for misclassifications
     - Adjust weights to minimize loss (gradient descent)
  3. Stop when convergence reached or max_iter exceeded

Code Example:
  >>> classifier.train_model(X_train, y_train)
  # Model learns:
  # - Which TF-IDF features strongly indicate each intent
  # - How to weight features for final prediction
"""


# =============================================================================
# STEP 5: MODEL EVALUATION
# =============================================================================

"""
Purpose:
  Assess model performance on unseen test data using multiple metrics.

Why Multiple Metrics?
  - Accuracy alone can be misleading with imbalanced data
  - Different metrics reveal different aspects of performance
  - Classification report shows per-class performance

Evaluation Metrics Explained:

  1. ACCURACY
     Formula: (TP + TN) / (TP + TN + FP + FN)
     Meaning: What percentage of predictions are correct?
     When useful: Balanced classes
     When misleading: Highly imbalanced classes
     Example: 92% accuracy might be poor if 95% of data is one class
  
  2. PRECISION (Per-class)
     Formula: TP / (TP + FP)
     Meaning: Of the emails I predicted as intent X, how many were correct?
     Interpretation: How trustworthy are my positive predictions?
     Example: "agendamento_reuniao" precision=0.88
       - When model says "schedule meeting", it's right 88% of the time
  
  3. RECALL (Per-class)
     Formula: TP / (TP + FN)
     Meaning: Of all emails that actually are intent X, how many did I find?
     Interpretation: How complete are my predictions?
     Example: "agendamento_reuniao" recall=0.91
       - I correctly identify 91% of actual scheduling requests
  
  4. F1-SCORE (Per-class)
     Formula: 2 × (Precision × Recall) / (Precision + Recall)
     Meaning: Harmonic mean of precision and recall
     Interpretation: Balanced measure when precision/recall trade off
     Example: F1=0.89 means good balance between precision and recall
  
  5. CONFUSION MATRIX
     Shows: Actual vs Predicted for all class pairs
     Reveals: Which classes get confused with each other
     Example:
       Predicted:    agend  cancel  discus  nao
       Actual:
       agend          38      1       2     1
       cancel          1     19       0     0
       discus          2      0      17     1
       nao             1      0       1     18
       
       Insight: "agendamento" sometimes confused with "discussao"
               (might combine these signals)

Expected Results for Good Model:
  - Accuracy: 80%+ on test set
  - Precision/Recall per class: 75%+
  - F1-scores balanced across classes
  - No class systematically misclassified

Code Example:
  >>> results = classifier.evaluate_model(X_test, y_test, test_labels)
  >>> print(f"Accuracy: {results['accuracy']:.2%}")
  Accuracy: 88.75%
  
  >>> print(results['classification_report'])
  # Shows detailed metrics for each intent class
"""


# =============================================================================
# STEP 6: MODEL PERSISTENCE
# =============================================================================

"""
Purpose:
  Save trained model and vectorizer to disk for later use without retraining.

Why Two Files?
  1. intent_classifier.joblib
     - The trained Logistic Regression model
     - Contains learned weights and bias values
     - Ready to make predictions
  
  2. tfidf_vectorizer.joblib
     - Fitted TfidfVectorizer with learned vocabulary
     - Contains the 5000 features discovered during training
     - Must be used to transform new emails before prediction
  
  Why Both Are Needed:
    - Vectorizer creates consistent feature space
    - If trained on dataset with "reunião" vocabulary,
      new emails must use same vocabulary
    - Can't mix vectorizers and models from different trainings

Storage Format:
  - joblib: Efficient binary format for scikit-learn objects
  - Preserves all internal state and parameters
  - Much smaller and faster than pickle for ML objects

Code Example:
  >>> classifier.save_model('models/')
  # Creates:
  # - models/intent_classifier.joblib
  # - models/tfidf_vectorizer.joblib
  
  # Later, load and use:
  >>> loaded = EmailIntentClassifier.load_model('models/')
  >>> X_new = loaded.vectorizer.transform(new_texts)
  >>> predictions = loaded.model.predict(X_new)
"""


# =============================================================================
# STEP 7: INFERENCE (USING THE TRAINED MODEL)
# =============================================================================

"""
Purpose:
  Use trained model to predict intent for new, unseen emails.

Inference Pipeline:
  1. Combine new email subject + body (same as training)
  2. Vectorize using saved TfidfVectorizer
     - MUST use same vectorizer as training
     - Creates same 5000-dimensional feature space
  3. Feed features to trained model
  4. Get prediction and confidence scores

Code Example:
  >>> from predict_intent import predict_email_intent
  >>> result = predict_email_intent(
  ...     subject="Reunião amanhã?",
  ...     body="Consegues vir às 14h?"
  ... )
  >>> print(result['intent'])
  agendamento_reuniao
  >>> print(f"{result['confidence']:.1%}")
  89.2%
  
  >>> print(result['probabilities'])
  {
    'agendamento_reuniao': 0.892,
    'cancelamento_reuniao': 0.067,
    'discussao_data': 0.035,
    'nao_reuniao': 0.006
  }

Decision Threshold:
  - Default: predict class with highest probability
  - Can adjust confidence threshold:
    if confidence > 0.90:
        accept_prediction()
    else:
        request_human_review()
"""


# =============================================================================
# KEY DESIGN DECISIONS
# =============================================================================

"""
1. Portuguese Accent Preservation
   - strip_accents=None in TfidfVectorizer
   - Reason: "reunião" ≠ "reuniao" in Portuguese
   - Semantic difference must be preserved

2. Balanced Class Weights
   - class_weight='balanced' in LogisticRegression
   - Reason: Intent distribution likely unbalanced
   - Prevents model from ignoring minority intents

3. Stratified Split
   - Maintains class distribution in train/test
   - Reliable evaluation metrics
   - Each set is representative of full dataset

4. Unigrams + Bigrams (1,2-gram)
   - Single words: "reunião", "data", "cliente"
   - Word pairs: "marcar reunião", "mudar data"
   - Captures context without huge feature space

5. Joblib for Persistence
   - Efficient for pickled objects
   - Handles sparse matrices well
   - Standard in scikit-learn ecosystem
"""


# =============================================================================
# TROUBLESHOOTING & IMPROVEMENTS
# =============================================================================

"""
Common Issues and Solutions:

1. Low Accuracy (< 75%)
   → Increase max_features (try 10000 or more)
   → Increase max_iter (try 5000 iterations)
   → Check data quality (missing values, encoding)
   → Check data volume (need ~100+ samples per class)

2. Class Imbalance Warnings
   → class_weight='balanced' addresses this
   → Collect more minority class samples
   → Consider data augmentation (paraphrasing)

3. Accent/Encoding Issues
   → Ensure JSON saved as UTF-8
   → Verify strip_accents=None (it's set correctly)
   → Check console encoding (Windows: chcp 65001)

4. Model File Size Too Large
   → Reduce max_features (5000 is already reasonable)
   → Use model compression (not essential for this size)

Next Steps to Improve Model:

1. Hyperparameter Tuning
   - GridSearchCV to find optimal parameters
   - Cross-validation for more robust evaluation

2. Feature Engineering
   - Add custom features (sender reputation, time of day, etc.)
   - TF-IDF with custom stop words for domain

3. Advanced Models
   - Naive Bayes (faster, competitive accuracy)
   - SVM (often better accuracy, slower training)
   - Neural networks (LSTM/BERT for deep learning)

4. Data Augmentation
   - Paraphrase existing emails
   - Generate synthetic examples
   - Back-translation (EN → PT → EN)

5. Ensemble Methods
   - Combine multiple models
   - Voting classifier for robustness
   - Stacking for improved accuracy
"""


# =============================================================================
# SUMMARY
# =============================================================================

"""
The complete pipeline:

Data → Load & Validate
   ↓
Split into Train/Test (80/20, stratified)
   ↓
Vectorize with TF-IDF (5000 features, 1-2 grams)
   ↓
Train Logistic Regression (balanced, 1000 iter)
   ↓
Evaluate on Test Set (accuracy, precision, recall, F1)
   ↓
Save Model + Vectorizer (joblib)
   ↓
Deploy: Load → Vectorize → Predict → Confidence

This baseline is production-ready and provides:
✓ Clear interpretability
✓ Fast predictions
✓ Reasonable accuracy for Portuguese email classification
✓ Minimal dependencies
✓ Easy to extend and improve
"""
