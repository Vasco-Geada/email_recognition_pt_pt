import json
import os
import logging
import sys
from pathlib import Path
from typing import Tuple, Dict, List

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmailIntentClassifier:
    """
    A text classification pipeline for email intent classification using TF-IDF and Logistic Regression.
    
    Attributes:
        vectorizer: TfidfVectorizer instance for text feature extraction
        model: Trained Logistic Regression classifier
        label2id: Mapping from label strings to class indices
        id2label: Mapping from class indices to label strings
    """
    
    def __init__(self):
        """Initialize the classifier components."""
        self.vectorizer = None
        self.model = None
        self.label2id = None
        self.id2label = None
        self.class_labels = [
            "agendamento_reuniao",
            "cancelamento_reuniao",
            "discussao_data",
            "nao_reuniao"
        ]
    
    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """
        Load dataset from a JSON file.
        
        Expected JSON format: List of objects with keys 'subject', 'body', 'label'
        Example:
            [
                {
                    "subject": "Meeting tomorrow",
                    "body": "Are you available?",
                    "label": "agendamento_reuniao"
                },
                ...
            ]
        
        Args:
            dataset_path: Path to the JSON dataset file
            
        Returns:
            pd.DataFrame: DataFrame with columns ['subject', 'body', 'label', 'text']
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            json.JSONDecodeError: If JSON is malformed
            ValueError: If required columns are missing
        """
        logger.info(f"Loading dataset from {dataset_path}")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            raise
        
        df = pd.DataFrame(data)
        
        # Validate required columns
        required_columns = {'subject', 'body', 'label'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")
        
        logger.info(f"Loaded {len(df)} samples")
        
        # Check for missing values
        missing_values = df[['subject', 'body', 'label']].isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values detected:\n{missing_values}")
            # Remove rows with missing values
            df = df.dropna(subset=['subject', 'body', 'label'])
            logger.info(f"Removed rows with missing values. Remaining: {len(df)} samples")
        
        # Combine subject and body into a single text field
        df['text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
        df['text'] = df['text'].str.strip()
        
        logger.info(f"Final dataset size: {len(df)} samples")
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets with stratification.
        
        Args:
            df: Input DataFrame with 'text' and 'label' columns
            test_size: Proportion of data for testing (default: 0.2 for 80/20 split)
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info(f"Splitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test (stratified)")
        
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['label']
        )
        
        logger.info(f"Training set: {len(train_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")
        
        return train_df, test_df
    
    def vectorize_text(
        self,
        train_texts: List[str],
        test_texts: List[str],
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorize text data using TF-IDF.
        
        Args:
            train_texts: Training text samples
            test_texts: Test text samples
            max_features: Maximum number of features (default: 5000)
            ngram_range: N-gram range (default: (1,2) for unigrams and bigrams)
            
        Returns:
            Tuple of (X_train_tfidf, X_test_tfidf) as sparse matrices
        """
        logger.info(f"Vectorizing text with TF-IDF (max_features={max_features}, ngrams={ngram_range})")
        
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=ngram_range,
            max_features=max_features,
            strip_accents=None,  # Preserve Portuguese accents
            min_df=1,
            max_df=1.0
        )
        
        X_train = self.vectorizer.fit_transform(train_texts)
        X_test = self.vectorizer.transform(test_texts)
        
        logger.info(f"Feature matrix shape: {X_train.shape}")
        logger.info(f"Vocabulary size: {len(self.vectorizer.get_feature_names_out())}")
        
        return X_train, X_test
    
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        max_iter: int = 1000,
        random_state: int = 42
    ) -> None:
        """
        Train a Logistic Regression classifier.
        
        Args:
            X_train: Training feature matrix
            y_train: Training labels (encoded as integers)
            max_iter: Maximum number of iterations (default: 1000)
            random_state: Random seed for reproducibility
        """
        logger.info("Training Logistic Regression model")
        logger.info(f"  - max_iter: {max_iter}")
        logger.info(f"  - class_weight: balanced")
        
        self.model = LogisticRegression(
            max_iter=max_iter,
            class_weight='balanced',
            random_state=random_state,
            solver='lbfgs',
            multi_class='multinomial'
        )
        
        self.model.fit(X_train, y_train)
        logger.info("Training completed successfully")
    
    def evaluate_model(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_test_labels: List[str]
    ) -> Dict:
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test feature matrix
            y_test: Test labels (encoded as integers)
            y_test_labels: Original test labels (strings)
            
        Returns:
            Dictionary with evaluation metrics (accuracy, classification_report, confusion_matrix)
        """
        logger.info("Evaluating model on test set")
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"MODEL EVALUATION RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"\nClassification Report:")
        logger.info("\n" + classification_report(y_test, y_pred, target_names=self.class_labels))
        logger.info(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\n{cm}\n")
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, target_names=self.class_labels, output_dict=True),
            'confusion_matrix': cm,
            'predictions': y_pred
        }
    
    def save_model(self, model_dir: str = 'models') -> None:
        """
        Save trained model and vectorizer to disk using joblib.
        
        Args:
            model_dir: Directory to save the model files
        """
        logger.info(f"Saving model to {model_dir}/")
        
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'intent_classifier.joblib')
        vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.joblib')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Vectorizer saved: {vectorizer_path}")
    
    @staticmethod
    def load_model(model_dir: str = 'models') -> 'EmailIntentClassifier':
        """
        Load a previously trained model and vectorizer.
        
        Args:
            model_dir: Directory containing the saved model files
            
        Returns:
            EmailIntentClassifier instance with loaded model and vectorizer
        """
        logger.info(f"Loading model from {model_dir}/")
        
        classifier = EmailIntentClassifier()
        
        model_path = os.path.join(model_dir, 'intent_classifier.joblib')
        vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.joblib')
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Model files not found in {model_dir}/")
        
        classifier.model = joblib.load(model_path)
        classifier.vectorizer = joblib.load(vectorizer_path)
        
        logger.info("Model loaded successfully")
        
        return classifier


def main(dataset_path: str = 'dataset/dataset.json', model_dir: str = 'models'):
    """
    Main training pipeline.
    
    Args:
        dataset_path: Path to the JSON dataset file
        model_dir: Directory to save the trained model
    """
    logger.info("Starting Email Intent Classification Pipeline")
    logger.info("=" * 60)
    
    try:
        # Initialize classifier
        classifier = EmailIntentClassifier()
        
        # Load dataset
        df = classifier.load_dataset(dataset_path)
        
        # Prepare data
        train_df, test_df = classifier.prepare_data(df)
        
        # Encode labels
        label_to_int = {label: idx for idx, label in enumerate(classifier.class_labels)}
        int_to_label = {idx: label for label, idx in label_to_int.items()}
        
        y_train = train_df['label'].map(label_to_int).values
        y_test = test_df['label'].map(label_to_int).values
        
        classifier.label2id = label_to_int
        classifier.id2label = int_to_label
        
        # Vectorize text
        X_train, X_test = classifier.vectorize_text(
            train_df['text'].values,
            test_df['text'].values,
            max_features=5000,
            ngram_range=(1, 2)
        )
        
        # Train model
        classifier.train_model(X_train, y_train, max_iter=1000)
        
        # Evaluate model
        results = classifier.evaluate_model(X_test, y_test, test_df['label'].values)
        
        # Save model
        classifier.save_model(model_dir)
        
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)
        
        return classifier, results
    
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    # Run the training pipeline
    classifier, results = main()
