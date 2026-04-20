"""
Enhanced Email Analysis Pipeline with Argument Extraction

Integrates:
1. Intent classification (from models/predict_intent.py)
2. Trigger extraction (from preprocessing/trigger_extraction.py)
3. Argument extraction (new!) - participants, time, location, topic

This creates a complete structured representation of meeting emails.
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from preprocessing.argument_extraction import ArgumentExtractor, ExtractedArguments
from preprocessing.trigger_extraction import TriggerExtractor


logger = logging.getLogger(__name__)


@dataclass
class StructuredEmailAnalysis:
    """
    Complete structured analysis of a meeting email.
    
    Combines intent classification, trigger extraction, and argument extraction.
    """
    # Original email
    email_subject: str
    email_body: str
    
    # Intent analysis
    predicted_intent: str
    intent_confidence: float
    
    # Trigger analysis
    trigger: Optional[str] = None
    trigger_method: Optional[str] = None
    
    # Arguments analysis
    participants: list = None
    time_expressions: list = None
    locations: list = None
    meeting_topics: list = None
    
    # Metadata
    analysis_timestamp: str = ""
    
    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.participants is None:
            self.participants = []
        if self.time_expressions is None:
            self.time_expressions = []
        if self.locations is None:
            self.locations = []
        if self.meeting_topics is None:
            self.meeting_topics = []
        if not self.analysis_timestamp:
            self.analysis_timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'email_metadata': {
                'subject': self.email_subject,
                'body_preview': self.email_body[:100],  # First 100 chars
                'analysis_timestamp': self.analysis_timestamp,
            },
            'intent_analysis': {
                'predicted_intent': self.predicted_intent,
                'confidence': self.intent_confidence,
            },
            'trigger_analysis': {
                'trigger': self.trigger,
                'method': self.trigger_method,
            },
            'extracted_arguments': {
                'participants': self.participants,
                'time_expressions': self.time_expressions,
                'locations': self.locations,
                'topics': self.meeting_topics,
            },
        }
    
    def actionable_summary(self) -> str:
        """Generate human-readable summary for decision-making."""
        lines = []
        
        # Intent
        intent_emoji = {
            'agendamento_reuniao': '📅',
            'cancelamento_reuniao': '❌',
            'reuniao_confirmada': '✅',
            'nao_reuniao': '📬',
        }
        emoji = intent_emoji.get(self.predicted_intent, '❔')
        lines.append(f"{emoji} {self.predicted_intent.replace('_', ' ').title()}")
        
        # Participants
        if self.participants:
            people = ', '.join(self.participants)
            lines.append(f"👥 With: {people}")
        
        # Time
        if self.time_expressions:
            times = ', '.join(self.time_expressions)
            lines.append(f"⏰ When: {times}")
        
        # Location
        if self.locations:
            places = ', '.join(self.locations)
            lines.append(f"📍 Where: {places}")
        
        # Topic
        if self.meeting_topics:
            topics = ', '.join(self.meeting_topics)
            lines.append(f"📎 About: {topics}")
        
        return '\n'.join(lines)


class EnhancedEmailAnalysisPipeline:
    """
    Complete email analysis pipeline with all components.
    
    Input: Raw email (subject + body)
    Output: StructuredEmailAnalysis with intent, trigger, and arguments
    """
    
    def __init__(self):
        """Initialize all pipeline components."""
        # Will be lazy-loaded on first use
        self._argument_extractor = None
        self._trigger_extractor = None
        
        logger.info("Enhanced Email Analysis Pipeline initialized")
    
    @property
    def argument_extractor(self):
        """Lazy-load argument extractor."""
        if self._argument_extractor is None:
            self._argument_extractor = ArgumentExtractor(model_name="pt_core_news_sm")
            logger.debug("Loaded ArgumentExtractor")
        return self._argument_extractor
    
    @property
    def trigger_extractor(self):
        """Lazy-load trigger extractor."""
        if self._trigger_extractor is None:
            self._trigger_extractor = TriggerExtractor()
            logger.debug("Loaded TriggerExtractor")
        return self._trigger_extractor
    
    def analyze(
        self,
        email_subject: str,
        email_body: str,
        predicted_intent: str,
        intent_confidence: float = 0.0,
    ) -> StructuredEmailAnalysis:
        """
        Perform complete email analysis.
        
        Args:
            email_subject: Email subject line
            email_body: Email body text
            predicted_intent: Intent from classifier (agendamento_reuniao, etc.)
            intent_confidence: Confidence score from intent classifier
            
        Returns:
            StructuredEmailAnalysis with all extracted information
        """
        logger.info(f"Analyzing email: {email_subject[:50]}...")
        
        # Step 1: Extract trigger
        trigger_result = self.trigger_extractor.extract_trigger(
            text=email_body,
            intent=predicted_intent
        )
        trigger = trigger_result.get("trigger") if trigger_result else None
        trigger_method = trigger_result.get("method") if trigger_result else None
        
        logger.debug(f"Extracted trigger: {trigger} (method: {trigger_method})")
        
        # Step 2: Extract arguments
        arguments = self.argument_extractor.extract(
            email_body=email_body,
            email_subject=email_subject,
            predicted_intent=predicted_intent,
            trigger=trigger or "",
        )
        
        logger.debug(
            f"Extracted arguments: "
            f"{len(arguments.participants)} participants, "
            f"{len(arguments.time_expressions)} times, "
            f"{len(arguments.locations)} locations, "
            f"{len(arguments.meeting_topics)} topics"
        )
        
        # Step 3: Convert argument spans to simple text list
        participants = [s.text for s in arguments.participants]
        times = [s.text for s in arguments.time_expressions]
        locations = [s.text for s in arguments.locations]
        topics = [s.text for s in arguments.meeting_topics]
        
        # Step 4: Create structured output
        analysis = StructuredEmailAnalysis(
            email_subject=email_subject,
            email_body=email_body,
            predicted_intent=predicted_intent,
            intent_confidence=intent_confidence,
            trigger=trigger,
            trigger_method=trigger_method,
            participants=participants,
            time_expressions=times,
            locations=locations,
            meeting_topics=topics,
        )
        
        logger.info(f"Email analysis complete: {analysis.predicted_intent}")
        return analysis
    
    def batch_analyze(
        self,
        emails: list,
        get_intent_fn,
    ) -> list:
        """
        Analyze multiple emails.
        
        Args:
            emails: List of email dicts with 'subject' and 'body'
            get_intent_fn: Function to predict intent from (subject, body)
                          Should return {'intent': str, 'confidence': float}
            
        Returns:
            List of StructuredEmailAnalysis objects
        """
        results = []
        
        for i, email in enumerate(emails, 1):
            logger.info(f"Processing email {i}/{len(emails)}")
            
            # Predict intent
            intent_result = get_intent_fn(email['subject'], email['body'])
            
            # Analyze
            analysis = self.analyze(
                email_subject=email['subject'],
                email_body=email['body'],
                predicted_intent=intent_result['intent'],
                intent_confidence=intent_result.get('confidence', 0.0),
            )
            
            results.append(analysis)
        
        return results


# Example usage functions
def example_basic_usage():
    """Example: Basic pipeline usage."""
    
    print("\n" + "="*80)
    print("EXAMPLE: Basic Enhanced Pipeline Usage")
    print("="*80)
    
    # Initialize pipeline
    pipeline = EnhancedEmailAnalysisPipeline()
    
    # Example email data
    email_subject = "Reunião - Projeto Q2"
    email_body = """
    Gostaria de agendar uma reunião com o João Silva e a Ana Costa para discutir 
    o cronograma do projeto Q2.
    
    Estou disponível amanhã à tarde na sala 304, ou sexta de manhã.
    
    Obrigado,
    Paulo
    """
    
    # Simulate intent prediction (from your classifier)
    predicted_intent = "agendamento_reuniao"
    intent_confidence = 0.92
    
    # Analyze email
    analysis = pipeline.analyze(
        email_subject=email_subject,
        email_body=email_body,
        predicted_intent=predicted_intent,
        intent_confidence=intent_confidence,
    )
    
    # Display results
    print("\n📊 Analysis Results:")
    print(analysis.actionable_summary())
    
    print("\n📋 Structured Data:")
    import json
    print(json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False))


def example_batch_processing():
    """Example: Processing multiple emails."""
    
    print("\n" + "="*80)
    print("EXAMPLE: Batch Email Processing")
    print("="*80)
    
    # Initialize pipeline
    pipeline = EnhancedEmailAnalysisPipeline()
    
    # Sample emails
    emails = [
        {
            "subject": "Reunião com cliente",
            "body": "Podemos agendar uma reunião com o cliente Silva para amanhã às 10h?"
        },
        {
            "subject": "Cancelo reunião",
            "body": "Infelizmente tenho que cancelar a reunião de sexta."
        },
        {
            "subject": "Confirmação",
            "body": "Confirmamos a reunião de segunda às 14h na sala 202."
        },
    ]
    
    # Mock intent predictor
    def mock_intent_predictor(subject, body):
        """Simulate intent classifier."""
        text = (subject + " " + body).lower()
        
        if "agendar" in text or "podemos" in text:
            return {"intent": "agendamento_reuniao", "confidence": 0.88}
        elif "cancela" in text or "cancelamento" in text:
            return {"intent": "cancelamento_reuniao", "confidence": 0.91}
        elif "confirma" in text or "confirmada" in text:
            return {"intent": "reuniao_confirmada", "confidence": 0.85}
        else:
            return {"intent": "nao_reuniao", "confidence": 0.70}
    
    # Process batch
    results = pipeline.batch_analyze(emails, mock_intent_predictor)
    
    # Display results
    print(f"\n✅ Processed {len(results)} emails\n")
    
    for i, analysis in enumerate(results, 1):
        print(f"Email {i}:")
        print(analysis.actionable_summary())
        print()


def example_integration_with_database():
    """Example: Saving analysis results to database/file."""
    
    print("\n" + "="*80)
    print("EXAMPLE: Integration with Storage")
    print("="*80)
    
    import json
    from datetime import datetime
    
    pipeline = EnhancedEmailAnalysisPipeline()
    
    # Analyze email
    email = {
        "subject": "Reunião de equipa",
        "body": """
        Reunião de retrospectiva amanhã de manhã às 10h.
        Presentes: João, Maria, Paulo, Ana
        Local: Sala 401
        Tópicos: Sprint review, impedimentos, próximos passos
        """
    }
    
    analysis = pipeline.analyze(
        email_subject=email["subject"],
        email_body=email["body"],
        predicted_intent="agendamento_reuniao",
        intent_confidence=0.89,
    )
    
    # Example: Save to JSON file
    output_file = "email_analysis_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis.to_dict(), f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved analysis to: {output_file}")
    print("\nPreview:")
    print(json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False)[:500])
    
    # Example: Convert to database record
    db_record = {
        "email_id": "email_001",
        "processed_at": analysis.analysis_timestamp,
        "intent": analysis.predicted_intent,
        "intent_confidence": analysis.intent_confidence,
        "participants": analysis.participants,
        "scheduled_times": analysis.time_expressions,
        "meeting_locations": analysis.locations,
        "meeting_topics": analysis.meeting_topics,
    }
    
    print("\n📦 Database Record:")
    print(json.dumps(db_record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run examples
    print("\n" + "#"*80)
    print("# ENHANCED EMAIL ANALYSIS PIPELINE - EXAMPLES")
    print("#"*80)
    
    try:
        example_basic_usage()
    except Exception as e:
        print(f"\n⚠️  Example 1 failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        example_batch_processing()
    except Exception as e:
        print(f"\n⚠️  Example 2 failed: {e}")
    
    try:
        example_integration_with_database()
    except Exception as e:
        print(f"\n⚠️  Example 3 failed: {e}")
    
    print("\n" + "#"*80)
    print("# QUICK START")
    print("#"*80)
    print("""
    To use in your code:
    
    from preprocessing.email_pipeline_enhanced import EnhancedEmailAnalysisPipeline
    
    pipeline = EnhancedEmailAnalysisPipeline()
    analysis = pipeline.analyze(
        email_subject="Subject",
        email_body="Body",
        predicted_intent="agendamento_reuniao",
        intent_confidence=0.92,
    )
    
    print(analysis.actionable_summary())
    print(analysis.to_dict())
    """)
