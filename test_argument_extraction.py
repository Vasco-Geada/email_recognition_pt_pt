"""
Test suite and examples for argument extraction module.

This module provides:
1. Test cases with annotated examples
2. Usage demonstrations
3. Expected outputs
4. Error case examples
"""

import json
from preprocessing.argument_extraction import ArgumentExtractor
from typing import Dict, List


class ArgumentExtractionTests:
    """Test cases for argument extraction."""
    
    # Test dataset: annotated examples
   
    with open("dataset/temp_emails.json", "r", encoding="utf-8") as f:
        TEST_EMAILS = json.load(f)
        
    @staticmethod
    def run_test(test_case: Dict) -> Dict:
        """Run a test case and return results."""
        extractor = ArgumentExtractor(model_name="pt_core_news_sm")
        
        result = extractor.extract_with_context(
            email_body=test_case["body"],
            email_subject=test_case["subject"],
            predicted_intent=test_case["intent"],
            trigger=test_case["trigger"] or "",
        )
        
        return {
            "test_id": test_case["id"],
            "subject": test_case["subject"],
            "extracted": result["extracted_arguments"],
            "expected": test_case["expected"],
            "metadata": result["metadata"],
        }
    
    @staticmethod
    def print_test_result(test_result: Dict) -> None:
        """Pretty-print test result."""
        print(f"\n{'='*80}")
        print(f"Test ID: {test_result['test_id']}")
        print(f"Subject: {test_result['subject']}")
        print(f"{'='*80}")
        
        # Print extracted vs expected
        extracted = test_result["extracted"]
        expected = test_result["expected"]
        
        print("\n📍 PARTICIPANTS")
        extracted_names = [s['text'] for s in extracted['participants']]
        print(f"  Extracted: {extracted_names}")
        print(f"  Expected:  {expected['participants']}")
        
        print("\n⏰ TEMPORAL EXPRESSIONS")
        extracted_times = [s['text'] for s in extracted['time_expressions']]
        print(f"  Extracted: {extracted_times}")
        print(f"  Expected:  {expected['times']}")
        
        print("\n📌 LOCATIONS")
        extracted_locs = [s['text'] for s in extracted['locations']]
        print(f"  Extracted: {extracted_locs}")
        print(f"  Expected:  {expected['locations']}")
        
        print("\n📎 TOPICS")
        extracted_topics = [s['text'] for s in extracted['topics']]
        print(f"  Extracted: {extracted_topics}")
        print(f"  Expected:  {expected['topics']}")


def example_detailed_extraction():
    """Example: Detailed extraction with confidence scores and spans."""
    
    print("\n" + "="*80)
    print("EXAMPLE: Detailed Extraction with Spans and Confidence")
    print("="*80)
    
    extractor = ArgumentExtractor(model_name="pt_core_news_sm")
    
    email_body = """
    Reunião com João Silva e ana@company.com para revisar o projeto Q1 
    na sala 405, bloco B, amanhã às 14:30.
    """
    
    result = extractor.extract_with_context(
        email_body=email_body,
        email_subject="Reunião de revisão - Q1",
        predicted_intent="agendamento_reuniao",
        include_confidence=True,
    )
    
    print("\nExtracted Arguments with Full Details:")
    print(json.dumps(result["extracted_arguments"], indent=2, ensure_ascii=False))
    
    print("\nInterpretation:")
    args = result["extracted_arguments"]
    
    for participant in args["participants"]:
        print(f"  - Participant: '{participant['text']}' ")
        print(f"    (confidence: {participant['confidence']:.2f}, method: {participant['extraction_method']})")
    
    for time_expr in args["time_expressions"]:
        print(f"  - Time: '{time_expr['text']}'")
        print(f"    (confidence: {time_expr['confidence']:.2f}, method: {time_expr['extraction_method']})")
    
    for location in args["locations"]:
        print(f"  - Location: '{location['text']}'")
        print(f"    (confidence: {location['confidence']:.2f}, method: {location['extraction_method']})")
    
    for topic in args["topics"]:
        print(f"  - Topic: '{topic['text']}'")
        print(f"    (confidence: {topic['confidence']:.2f}, method: {topic['extraction_method']})")


def example_temporal_patterns():
    """Example: Various Portuguese temporal patterns."""
    
    print("\n" + "="*80)
    print("EXAMPLE: Portuguese Temporal Expression Variants")
    print("="*80)
    
    extractor = ArgumentExtractor(model_name="pt_core_news_sm")
    
    temporal_examples = [
        ("Amanhã às 15h", "Tomorrow at 3 PM"),
        ("Sexta-feira próxima de tarde", "Next Friday afternoon"),
        ("Depois de almoço", "After lunch"),
        ("De manhã cedo", "Early morning"),
        ("5 de março de 2024", "March 5, 2024"),
        ("Próxima segunda", "Next Monday"),
        ("Daqui a 3 dias", "In 3 days"),
        ("09:30", "9:30 AM"),
        ("Hoje às 14 horas", "Today at 14:00"),
        ("Esta semana", "This week"),
    ]
    
    print("\nTemporal Pattern Recognition:")
    for email_snippet, translation in temporal_examples:
        result = extractor.temporal_extractor.extract(email_snippet)
        extracted = [span.text for span in result]
        print(f"  '{email_snippet}' ({translation})")
        print(f"    → {extracted if extracted else 'No match'}")


def example_location_detection():
    """Example: Location and room detection."""
    
    print("\n" + "="*80)
    print("EXAMPLE: Location and Room Detection")
    print("="*80)
    
    extractor = ArgumentExtractor(model_name="pt_core_news_sm")
    
    location_examples = [
        "Sala 203, 1º andar",
        "Escritório nº 5, bloco A",
        "Auditório B",
        "Laboratório de IA",
        "Rua da Prata, 50, Lisboa",
        "2º piso, edifício 1",
        "Anfiteatro do campus",
    ]
    
    print("\nLocation Recognition:")
    for location_snippet in location_examples:
        result = extractor.location_extractor.extract(location_snippet)
        extracted = [span.text for span in result]
        print(f"  '{location_snippet}'")
        print(f"    → {extracted if extracted else 'No match'}")


def example_error_cases():
    """Example: Known error cases and limitations."""
    
    print("\n" + "="*80)
    print("EXAMPLE: Known Limitations and Error Cases")
    print("="*80)
    
    extractor = ArgumentExtractor(model_name="pt_core_news_sm")
    
    error_cases = [
        {
            "name": "Negation (captures but shouldn't apply)",
            "email": "Não é amanhã, mas sim quinta-feira.",
            "issue": "Extracts both 'amanhã' and 'quinta-feira'",
            "mitigation": "Add negation context detection",
        },
        {
            "name": "Ambiguous informal time",
            "email": "De tarde na sala de espera.",
            "issue": "'sala de espera' (waiting room) extracts as location",
            "mitigation": "Intent filtering: agendamento → prioritize meeting rooms",
        },
        {
            "name": "Email signature pollution",
            "email": "Reunião com Jose.\n\n--\nFroman: Maria\nCC: Paulo, Ana, Bruno",
            "issue": "All names extracted including signature",
            "mitigation": "Pre-process to remove signature (look for '--')",
        },
        {
            "name": "Abbreviations",
            "email": "Reunião com JS e MPS.",
            "issue": "NER fails on abbreviations (JS, MPS)",
            "mitigation": "Company directory lookup (advanced)",
        },
        {
            "name": "Relative date without anchor",
            "email": "Próxima semana pode ser?",
            "issue": "Relative date needs current date for normalization",
            "mitigation": "Store extraction date; provide relative/absolute both",
        },
    ]
    
    print("\nKnown Issues:")
    for case in error_cases:
        print(f"\n  ❌ {case['name']}")
        print(f"     Email: '{case['email']}'")
        print(f"     Issue: {case['issue']}")
        print(f"     Mitigation: {case['mitigation']}")


def example_pipeline_integration():
    """Example: Full pipeline with intent, trigger, and arguments."""
    
    print("\n" + "="*80)
    print("EXAMPLE: Full Pipeline Integration")
    print("="*80)
    
    from preprocessing.trigger_extraction import TriggerExtractor
    
    email_body = """
    Preciso de agendar uma reunião com o João para discutir o cronograma 
    do projeto Q2. Podemos fazer amanhã à tarde na sala 304?
    """
    
    email_subject = "Reunião urgente - Cronograma Q2"
    
    print(f"\nInput Email:")
    print(f"  Subject: {email_subject}")
    print(f"  Body: {email_body.strip()}")
    
    # Step 1: Intent classification (simulated)
    predicted_intent = "agendamento_reuniao"
    intent_confidence = 0.94
    
    # Step 2: Trigger extraction (simulated)
    trigger_extractor = TriggerExtractor()
    trigger_result = trigger_extractor.extract_trigger(
        text=email_body,
        intent=predicted_intent
    )
    trigger = trigger_result.get("trigger", "") if trigger_result else ""
    
    # Step 3: Argument extraction
    arg_extractor = ArgumentExtractor(model_name="pt_core_news_sm")
    arguments = arg_extractor.extract(
        email_body=email_body,
        email_subject=email_subject,
        predicted_intent=predicted_intent,
        trigger=trigger,
    )
    
    print(f"\n✅ Intent Classification:")
    print(f"  Intent: {predicted_intent}")
    print(f"  Confidence: {intent_confidence:.2%}")
    print(f"  Trigger: '{trigger}'")
    
    print(f"\n📊 Extracted Arguments:")
    print(f"  Participants: {[p.text for p in arguments.participants]}")
    print(f"  Times: {[t.text for t in arguments.time_expressions]}")
    print(f"  Locations: {[l.text for l in arguments.locations]}")
    print(f"  Topics: {[t.text for t in arguments.topics]}")
    
    print(f"\n🎯 Actionable Summary:")
    print(f"  Scheduled with: {', '.join([p.text for p in arguments.participants]) or 'Unknown'}")
    if arguments.time_expressions:
        print(f"  When: {', '.join([t.text for t in arguments.time_expressions])}")
    if arguments.locations:
        print(f"  Where: {', '.join([l.text for l in arguments.locations])}")
    if arguments.topics:
        print(f"  About: {', '.join([t.text for t in arguments.topics])}")


if __name__ == "__main__":
    # Run comprehensive examples
    
    print("\n" + "#"*80)
    print("# ARGUMENT EXTRACTION - TEST SUITE AND EXAMPLES")
    print("#"*80)
    
    # Example 1: Temporal patterns
    try:
        example_temporal_patterns()
    except Exception as e:
        print(f"\n⚠️  Temporal patterns example failed: {e}")
    
    # Example 2: Location detection
    try:
        example_location_detection()
    except Exception as e:
        print(f"\n⚠️  Location detection example failed: {e}")
    
    # Example 3: Error cases
    try:
        example_error_cases()
    except Exception as e:
        print(f"\n⚠️  Error cases example failed: {e}")
    
    # Example 4: Detailed extraction with spans
    try:
        example_detailed_extraction()
    except Exception as e:
        print(f"\n⚠️  Detailed extraction example failed: {e}")
    
    # Example 5: Full pipeline
    try:
        example_pipeline_integration()
    except Exception as e:
        print(f"\n⚠️  Pipeline integration example failed: {e}")
    
    # Run test suite (if spaCy model available)
    print("\n" + "#"*80)
    print("# TEST SUITE - Annotated Examples")
    print("#"*80)
    
    # Note: Requires pt_core_news_sm model
    print("\n⚠️  Note: Full test suite requires spaCy PT model:")
    print("    python -m spacy download pt_core_news_sm")
    print("\nTo run tests, call:")
    print("    for test_case in ArgumentExtractionTests.TEST_EMAILS:")
    print("        result = ArgumentExtractionTests.run_test(test_case)")
    print("        ArgumentExtractionTests.print_test_result(result)")
