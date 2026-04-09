"""
Email Analysis Pipeline: Integrated Intent Classification + Trigger Extraction

This module provides a complete, production-ready pipeline that combines:
1. Intent classification (from predict_intent.py)
2. Trigger extraction (from trigger_extraction.py)

Creates structured, actionable insights from raw email text.
"""

import logging
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

from preprocessing.trigger_extraction import TriggerExtractor


logger = logging.getLogger(__name__)


@dataclass
class EmailAnalysis:
    """
    Structured result of email analysis.
    
    Attributes:
        text_preview: First 100 chars of email for reference
        intent: Predicted intent class
        intent_confidence: Confidence score (0-1)
        trigger: Extracted trigger word/phrase
        trigger_method: How trigger was found ('exact', 'regex', 'lemma', or 'none')
        requires_action: Whether this email needs human action
        priority: Priority level ('high', 'medium', 'low')
        timestamp: When analysis was performed
        actionable_summary: Human-readable summary for decision-making
    """
    text_preview: str
    intent: str
    intent_confidence: float
    trigger: Optional[str]
    trigger_method: Optional[str]
    requires_action: bool
    priority: str
    timestamp: str
    actionable_summary: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json_serializable(self) -> Dict:
        """Convert to JSON-serializable format."""
        return self.to_dict()


class EmailAnalysisPipeline:
    """
    End-to-end email analysis pipeline.
    
    Combines intent classification and trigger extraction into a single,
    easy-to-use interface.
    
    Usage:
        pipeline = EmailAnalysisPipeline(intent_classifier)
        analysis = pipeline.analyze(email_text)
        
        if analysis.requires_action:
            process_email(analysis)
    """
    
    # Email intents that require user action
    ACTION_REQUIRED_INTENTS = {
        "agendamento_reuniao",      # Schedule action needed
        "cancelamento_reuniao",      # Cancellation action needed
        "reuniao_confirmada",            # Decision needed
    }
    
    # No action required for these
    INFO_ONLY_INTENTS = {
        "nao_reuniao",               # FYI emails
    }
    
    # Priority mapping: intent → priority level
    INTENT_PRIORITY = {
        "agendamento_reuniao": "high",      # Scheduling often time-sensitive
        "cancelamento_reuniao": "high",     # Cancellations need quick response
        "reuniao_confirmada": "medium",        # Data discussion less urgent
        "nao_reuniao": "low",              # Info emails are non-urgent
    }
    
    def __init__(
        self,
        intent_classifier,
        use_trigger_lemmatization: bool = False
    ):
        """
        Initialize the pipeline.
        
        Args:
            intent_classifier: Model with predict() method
                              Input: email text
                              Output: dict with intent→probability
            use_trigger_lemmatization: Enable spaCy lemmatization for triggers
        """
        self.intent_classifier = intent_classifier
        self.trigger_extractor = TriggerExtractor(
            use_lemmatization=use_trigger_lemmatization
        )
        logger.info(f"Pipeline initialized")
        logger.info(f"  Lemmatization: {use_trigger_lemmatization}")
    
    def _get_priority(self, intent: str, confidence: float) -> str:
        """
        Determine email priority based on intent and confidence.
        
        Args:
            intent: Predicted intent
            confidence: Intent prediction confidence (0-1)
            
        Returns:
            Priority level: 'high', 'medium', or 'low'
        """
        base_priority = self.INTENT_PRIORITY.get(intent, "low")
        
        # Reduce priority if confidence is low (uncertain predictions)
        if confidence < 0.6:
            if base_priority == "high":
                return "medium"
            elif base_priority == "medium":
                return "low"
        
        return base_priority
    
    def _get_actionable_summary(
        self,
        intent: str,
        trigger: Optional[str],
        trigger_method: Optional[str]
    ) -> str:
        """
        Generate human-readable action summary.
        
        Args:
            intent: Predicted intent
            trigger: Extracted trigger (if any)
            trigger_method: How trigger was found
            
        Returns:
            Summary string for decision-making
        """
        summaries = {
            "agendamento_reuniao": (
                f"📅 SCHEDULE MEETING OR DISCUSS DATE/TIME"
                f" | Trigger: {trigger} ({trigger_method})"
                if trigger else "📅 SCHEDULE MEETING (no clear trigger)"
            ),
            "cancelamento_reuniao": (
                f"❌ CANCELLATION NOTICE"
                f" | Trigger: {trigger} ({trigger_method})"
                if trigger else "❌ CANCELLATION NOTICE (confirm via email)"
            ),
            "reuniao_confirmada": (
                f"🗓️ CONFIRM MEETING OR DATE/TIME"
                f" | Trigger: {trigger} ({trigger_method})"
                if trigger else "🗓️ DISCUSS DATE/TIME (no specific trigger)"
            ),
            "nao_reuniao": (
                "ℹ️ INFO ONLY (no action needed)"
            ),
        }
        
        return summaries.get(intent, "UNKNOWN INTENT")
    
    def analyze(
        self,
        email_text: str,
        return_dict: bool = False
    ) -> Optional[EmailAnalysis]:
        """
        Analyze an email end-to-end.
        
        Process:
        1. Classify intent
        2. Extract trigger
        3. Determine priority
        4. Generate summary
        5. Return structured result
        
        Args:
            email_text: Full email body
            return_dict: If True, return as dict instead of EmailAnalysis
            
        Returns:
            EmailAnalysis object (or dict if return_dict=True)
            
        Example:
            >>> analysis = pipeline.analyze(email_text)
            >>> print(analysis.intent)
            'agendamento_reuniao'
            >>> print(analysis.actionable_summary)
            '📅 SCHEDULE MEETING | Trigger: agendar (exact)'
        """
        if not email_text or not email_text.strip():
            logger.warning("Empty email text provided")
            return None
        
        # Step 1: Classify intent
        try:
            intent_probs = self.intent_classifier.predict(email_text)
            
            if not intent_probs:
                logger.error("Intent classifier returned no predictions")
                return None
            
            # Get top intent and confidence
            predicted_intent = max(intent_probs, key=intent_probs.get)
            confidence = intent_probs[predicted_intent]
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return None
        
        # Step 2: Extract trigger
        try:
            trigger_result = self.trigger_extractor.extract_trigger(
                email_text,
                predicted_intent
            )
            trigger = trigger_result["trigger"] if trigger_result else None
            trigger_method = trigger_result["method"] if trigger_result else None
        except Exception as e:
            logger.error(f"Trigger extraction failed: {e}")
            trigger = None
            trigger_method = None
        
        # Step 3: Determine if action is required
        requires_action = (
            predicted_intent in self.ACTION_REQUIRED_INTENTS
        )
        
        # Step 4: Determine priority
        priority = self._get_priority(predicted_intent, confidence)
        
        # Step 5: Generate summary
        actionable_summary = self._get_actionable_summary(
            predicted_intent,
            trigger,
            trigger_method
        )
        
        # Step 6: Create result
        analysis = EmailAnalysis(
            text_preview=email_text[:100] + ("..." if len(email_text) > 100 else ""),
            intent=predicted_intent,
            intent_confidence=float(confidence),
            trigger=trigger,
            trigger_method=trigger_method,
            requires_action=requires_action,
            priority=priority,
            timestamp=datetime.now().isoformat(),
            actionable_summary=actionable_summary
        )
        
        logger.info(f"Email analyzed: {predicted_intent} (conf={confidence:.2f})")
        
        if return_dict:
            return analysis.to_dict()
        
        return analysis
    
    def analyze_batch(
        self,
        email_texts: List[str],
        return_dicts: bool = False
    ) -> List[Optional[EmailAnalysis]]:
        """
        Analyze multiple emails.
        
        Args:
            email_texts: List of email bodies
            return_dicts: If True, return dicts instead of EmailAnalysis objects
            
        Returns:
            List of EmailAnalysis results (or dicts)
        """
        results = []
        
        for i, text in enumerate(email_texts):
            try:
                analysis = self.analyze(text, return_dict=return_dicts)
                results.append(analysis)
            except Exception as e:
                logger.error(f"Failed to analyze email {i}: {e}")
                results.append(None)
        
        return results
    
    def get_action_items(
        self,
        analyses: List[EmailAnalysis]
    ) -> Dict[str, List[EmailAnalysis]]:
        """
        Organize analyses by action type for processing.
        
        Args:
            analyses: List of EmailAnalysis results
            
        Returns:
            Dict organized by intent:
            {
                'agendamento_reuniao': [...],
                'cancelamento_reuniao': [...],
                'reuniao_confirmada': [...],
                'nao_reuniao': [...]
            }
        """
        action_items = {
            intent: [] for intent in self.INTENT_PRIORITY.keys()
        }
        
        for analysis in analyses:
            if analysis and analysis.intent in action_items:
                action_items[analysis.intent].append(analysis)
        
        # Sort by priority (high → low)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        for intent in action_items:
            action_items[intent].sort(
                key=lambda a: priority_order.get(a.priority, 3)
            )
        
        return action_items
    
    def get_stats(
        self,
        analyses: List[EmailAnalysis]
    ) -> Dict:
        """
        Get statistics about analyzed emails.
        
        Args:
            analyses: List of EmailAnalysis results
            
        Returns:
            Statistics dictionary
        """
        valid_analyses = [a for a in analyses if a]
        
        if not valid_analyses:
            return {"total": 0, "analyzed": 0}
        
        intent_counts = {}
        trigger_hits = 0
        high_priority_count = 0
        
        for analysis in valid_analyses:
            intent = analysis.intent
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            if analysis.trigger:
                trigger_hits += 1
            
            if analysis.priority == "high":
                high_priority_count += 1
        
        return {
            "total": len(analyses),
            "analyzed": len(valid_analyses),
            "by_intent": intent_counts,
            "trigger_hit_rate": trigger_hits / len(valid_analyses) if valid_analyses else 0,
            "high_priority_count": high_priority_count,
            "action_required_count": len([
                a for a in valid_analyses if a.requires_action
            ]),
        }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================
"""
Example 1: Single Email Analysis
─────────────────────────────────

from models.predict_intent import IntentClassifier
from preprocessing.email_pipeline import EmailAnalysisPipeline

# Setup
classifier = IntentClassifier()
pipeline = EmailAnalysisPipeline(classifier, use_trigger_lemmatization=True)

# Analyze
email = "Olá, gostaria de agendar uma reunião para próxima semana."
analysis = pipeline.analyze(email)

# Use result
print(f"Intent: {analysis.intent}")
print(f"Trigger: {analysis.trigger}")
print(f"Action Needed: {analysis.requires_action}")
print(f"Priority: {analysis.priority}")
print(f"\nSummary: {analysis.actionable_summary}")


Example 2: Batch Processing with Organization
──────────────────────────────────────────────

import pandas as pd

# Load emails
df = pd.read_csv('emails.csv')
emails = df['body'].tolist()

# Analyze all
analyses = pipeline.analyze_batch(emails)

# Organize by action type
action_items = pipeline.get_action_items(analyses)

# Process scheduling emails first
for email_analysis in action_items['agendamento_reuniao']:
    process_schedule_request(email_analysis)

# Then cancellations
for email_analysis in action_items['cancelamento_reuniao']:
    process_cancellation(email_analysis)


Example 3: Dashboard/Monitoring
───────────────────────────────

# Analyze batch
analyses = pipeline.analyze_batch(email_texts)

# Get statistics
stats = pipeline.get_stats(analyses)

# Dashboard display
print(f"Emails Analyzed: {stats['total']}")
print(f"Trigger Success Rate: {stats['trigger_hit_rate']*100:.1f}%")
print(f"\nBy Intent:")
for intent, count in stats['by_intent'].items():
    print(f"  {intent}: {count}")

print(f"\nActionable Items:")
print(f"  High Priority: {stats['high_priority_count']}")
print(f"  Total Requiring Action: {stats['action_required_count']}")


Example 4: JSON Export for API
──────────────────────────────

import json

# Analyze
analysis = pipeline.analyze(email_text)

# Export as JSON
json_output = json.dumps(analysis.to_json_serializable(), indent=2)

# Response
{
  "text_preview": "Olá, gostaria de agendar uma reunião...",
  "intent": "agendamento_reuniao",
  "intent_confidence": 0.95,
  "trigger": "agendar",
  "trigger_method": "exact",
  "requires_action": true,
  "priority": "high",
  "timestamp": "2026-03-31T14:32:00.123456",
  "actionable_summary": "📅 SCHEDULE MEETING | Trigger: agendar (exact)"
}
"""
