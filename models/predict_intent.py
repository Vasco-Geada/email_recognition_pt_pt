import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.train_intent import EmailIntentClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def predict_email_intent(subject: str, body: str, model_dir: str = 'models') -> dict:
    """
    Predict the intent of an email given its subject and body.
    
    Args:
        subject: Email subject line
        body: Email body/content
        model_dir: Directory where the trained model is saved
        
    Returns:
        Dictionary with keys:
        - intent: Predicted intent class
        - confidence: Prediction confidence (0-1)
        - probabilities: Probability for each class
    """
    try:
        classifier = EmailIntentClassifier.load_model(model_dir)
        
        email_text = f"{subject} {body}".strip()
        
        X = classifier.vectorizer.transform([email_text])
        
        # Get prediction and probabilities
        prediction = classifier.model.predict(X)[0]
        probabilities = classifier.model.predict_proba(X)[0]
        
        # Get the predicted intent
        intent = classifier.class_labels[prediction]
        confidence = probabilities[prediction]
        
        # Map all probabilities to class names
        prob_dict = {
            label: float(prob)
            for label, prob in zip(classifier.class_labels, probabilities)
        }
        
        return {
            'intent': intent,
            'confidence': float(confidence),
            'probabilities': prob_dict
        }
    
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        logger.error("Please run 'python models/train_intent.py' first to train the model.")
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise


def main():
    """Run example email intent predictions."""
    
    logger.info("Email Intent Classification - Inference Example")
    logger.info("=" * 60)
    
    # Example emails for testing
    test_emails = [
        {
            "subject": "Reunião com o cliente",
            "body": "Gostaria de agendar uma reunião para discutir o projeto."
        },
        {
            "subject": "Cancelamento",
            "body": "Infelizmente tenho que cancelar a reunião de amanhã."
        },
        {
            "subject": "Qual é a melhor data?",
            "body": "Qual é a melhor data para nos reunirmos? Terça ou quarta?"
        },
        {
            "subject": "Relatório mensal",
            "body": "Aqui está o relatório mensal de vendas. Por favor, reveja e envie seus comentários."
        }
    ]
    
    for i, email in enumerate(test_emails, 1):
        logger.info(f"\n{'─' * 60}")
        logger.info(f"Example {i}:")
        logger.info(f"Subject: {email['subject']}")
        logger.info(f"Body: {email['body']}")
        
        try:
            result = predict_email_intent(email['subject'], email['body'])
            
            logger.info(f"\nPredicted Intent: {result['intent']}")
            logger.info(f"Confidence: {result['confidence']:.2%}")
            logger.info(f"\nProbability Distribution:")
            for intent, prob in result['probabilities'].items():
                bar = "█" * int(prob * 30)
                logger.info(f"  {intent:25} {prob:.2%} {bar}")
        
        except Exception as e:
            logger.error(f"Failed to predict: {e}")
    
    logger.info(f"\n{'=' * 60}")
    logger.info("Inference example completed!")


if __name__ == '__main__':
    main()
