"""
Example Usage and Tests for TriggerExtractor Module

Demonstrates:
1. Basic trigger extraction
2. Batch processing
3. Lemmatization with spaCy
4. Integration with intent classifier
5. Error handling
"""

from preprocessing.trigger_extraction import TriggerExtractor, EmailIntent


def example_basic_extraction():
    """Basic single-sample trigger extraction."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Trigger Extraction")
    print("=" * 70)
    
    extractor = TriggerExtractor(use_lemmatization=False)
    
    # Example 1: Agendamento
    text1 = "Olá, gostaria de agendar uma reunião para próxima semana. Quando você está disponível?"
    result1 = extractor.extract_trigger(text1, "agendamento_reuniao")
    print(f"\nText: {text1}")
    print(f"Intent: agendamento_reuniao")
    print(f"Result: {result1}\n")
    
    # Example 2: Cancelamento
    text2 = "Infelizmente, não vou poder comparecer. Terei que cancelar a reunião."
    result2 = extractor.extract_trigger(text2, "cancelamento_reuniao")
    print(f"Text: {text2}")
    print(f"Intent: cancelamento_reuniao")
    print(f"Result: {result2}\n")
    
    # Example 3: Discussão de Data
    text3 = "Qual dia você sugere para a próxima reunião de equipa?"
    result3 = extractor.extract_trigger(text3, "discussao_data")
    print(f"Text: {text3}")
    print(f"Intent: discussao_data")
    print(f"Result: {result3}\n")
    
    # Example 4: No Trigger Found
    text4 = "O relatório de vendas está anexado em arquivo."
    result4 = extractor.extract_trigger(text4, "agendamento_reuniao")
    print(f"Text: {text4}")
    print(f"Intent: agendamento_reuniao")
    print(f"Result: {result4}\n")


def example_batch_processing():
    """Process multiple emails at once."""
    print("=" * 70)
    print("EXAMPLE 2: Batch Processing")
    print("=" * 70)
    
    extractor = TriggerExtractor(use_lemmatization=False)
    
    texts = [
        "Vamos marcar a reunião para quinta-feira?",
        "Preciso cancelar nosso encontro de amanhã.",
        "Qual horário você prefere?",
        "A proposta está pronta para discussão.",
    ]
    
    intents = [
        "agendamento_reuniao",
        "cancelamento_reuniao",
        "discussao_data",
        "nao_reuniao",
    ]
    
    results = extractor.extract_trigger_batch(texts, intents)
    
    for text, intent, result in zip(texts, intents, results):
        status = "✓ Found" if result else "✗ Not found"
        trigger = result["trigger"] if result else "N/A"
        method = result["method"] if result else "N/A"
        print(f"\n{status}")
        print(f"  Text:    {text}")
        print(f"  Intent:  {intent}")
        print(f"  Trigger: {trigger} ({method})")


def example_regex_matching():
    """Demonstrate regex pattern matching for inflections."""
    print("=" * 70)
    print("EXAMPLE 3: Regex Pattern Matching (Verb Conjugations)")
    print("=" * 70)
    
    extractor = TriggerExtractor(use_lemmatization=False)
    
    # Test different verb conjugations
    conjugations = [
        "Vou agendar a reunião.",           # infinitive
        "Agendei para amanhã.",              # past
        "Agendo agora mesmo.",               # present
        "Agendarei na próxima semana.",       # future
        "Estou agendando a reunião.",        # progressive
    ]
    
    for text in conjugations:
        result = extractor.extract_trigger(text, "agendamento_reuniao")
        trigger = result["trigger"] if result else "Not found"
        print(f"  {text:40} → {trigger}")


def example_optional_lemmatization():
    """Demonstrate spaCy lemmatization (if available)."""
    print("=" * 70)
    print("EXAMPLE 4: Optional Lemmatization with spaCy")
    print("=" * 70)
    
    # Try with lemmatization enabled
    extractor = TriggerExtractor(use_lemmatization=True)
    
    if not extractor.use_lemmatization:
        print("  ⚠️  spaCy Portuguese model not installed.")
        print("  Install with: pip install spacy")
        print("  Download with: python -m spacy download pt_core_news_sm")
        return
    
    # These would benefit from lemmatization
    texts = [
        "Agendamento de reunião para o dia 15.",
        "Cancelando o encontro de amanhã.",
        "Adiei o compromisso para próxima semana.",
    ]
    
    print("\n  With Lemmatization Enabled:")
    for text in texts:
        trigger_result = extractor.extract_trigger(text, "agendamento_reuniao")
        method = trigger_result["method"] if trigger_result else "No trigger"
        print(f"  {text:50} → {method}")


def example_intent_aware_extraction():
    """Show why intent awareness matters."""
    print("=" * 70)
    print("EXAMPLE 5: Why Intent-Aware Extraction Matters")
    print("=" * 70)
    
    extractor = TriggerExtractor(use_lemmatization=False)
    
    # Same word, different trigger status based on intent
    text = "Qual dia você prefere para agendar a reunião?"
    
    result_agend = extractor.extract_trigger(text, "agendamento_reuniao")
    result_discus = extractor.extract_trigger(text, "discussao_data")
    
    print(f"\nText: {text}")
    print(f"\n  As agendamento_reuniao:")
    print(f"    Trigger: {result_agend['trigger'] if result_agend else 'None'}")
    print(f"    (Action: SCHEDULE the meeting)")
    
    print(f"\n  As discussao_data:")
    print(f"    Trigger: {result_discus['trigger'] if result_discus else 'None'}")
    print(f"    (Action: DISCUSS the date)")
    
    print("\n  → Same sentence, different meaning based on intent!")


def example_integration_with_classifier():
    """
    Show integration with intent classifier.
    Note: Requires models.predict_intent module to be available
    """
    print("=" * 70)
    print("EXAMPLE 6: Integration with Intent Classifier")
    print("=" * 70)
    
    print("\nPseudocode integration:")
    print("""
    from models.predict_intent import IntentClassifier
    from preprocessing.trigger_extraction import TriggerExtractor
    
    # Pipeline setup
    classifier = IntentClassifier()
    extractor = TriggerExtractor(use_lemmatization=True)
    
    # Process email
    email_body = "Gostaria de agendar uma reunião para próxima quinta."
    
    # Step 1: Predict intent
    intent_probs = classifier.predict(email_body)
    predicted_intent = max(intent_probs, key=intent_probs.get)
    # → "agendamento_reuniao" with probability 0.95
    
    # Step 2: Extract trigger
    trigger_result = extractor.extract_trigger(email_body, predicted_intent)
    # → {'trigger': 'agendar', 'method': 'exact', 'intent': 'agendamento_reuniao'}
    
    # Step 3: Create structured output
    email_analysis = {
        'intent': predicted_intent,
        'intent_confidence': intent_probs[predicted_intent],
        'trigger': trigger_result['trigger'],
        'trigger_method': trigger_result['method'],
        'requires_action': predicted_intent in [
            'agendamento_reuniao',
            'cancelamento_reuniao'
        ]
    }
    
    # Use structured output
    if email_analysis['requires_action']:
        notify_user(
            f"ACTION REQUIRED: {email_analysis['intent']}",
            f"Trigger found: {email_analysis['trigger']}"
        )
    """)


def example_error_handling():
    """Demonstrate error handling."""
    print("=" * 70)
    print("EXAMPLE 7: Error Handling")
    print("=" * 70)
    
    extractor = TriggerExtractor()
    
    # Invalid intent
    print("\n1. InvalidIntent:")
    try:
        extractor.extract_trigger("Some text", "invalid_intent")
    except ValueError as e:
        print(f"   ✓ Caught error: {e}")
    
    # Empty text
    print("\n2. Empty Text:")
    result = extractor.extract_trigger("", "agendamento_reuniao")
    print(f"   ✓ Gracefully handled: {result}")
    
    # Mismatched batch lengths
    print("\n3. Mismatched Batch Dimensions:")
    try:
        extractor.extract_trigger_batch(["text1", "text2"], ["intent1"])
    except ValueError as e:
        print(f"   ✓ Caught error: {e}")


def example_performance_characteristics():
    """Show performance characteristics."""
    print("=" * 70)
    print("EXAMPLE 8: Performance & Complexity")
    print("=" * 70)
    
    import time
    
    extractor = TriggerExtractor(use_lemmatization=False)
    
    # Time a single extraction
    text = "Gostaria de agendar uma reunião para próxima semana, se possível."
    
    start = time.time()
    for _ in range(1000):
        extractor.extract_trigger(text, "agendamento_reuniao")
    elapsed = time.time() - start
    
    print(f"\nLexical Extraction (no lemmatization):")
    print(f"  1000 extractions: {elapsed:.3f}s")
    print(f"  Per extraction: {elapsed/1000*1000:.2f}ms")
    print(f"  Throughput: {1000/elapsed:.0f} emails/sec")
    
    print(f"\nWith lemmatization enabled:")
    print(f"  ~10-50x slower (depends on text length)")
    print(f"  ~20-100 emails/sec (spaCy parsing overhead)")


def main():
    """Run all examples."""
    example_basic_extraction()
    example_batch_processing()
    example_regex_matching()
    example_optional_lemmatization()
    example_intent_aware_extraction()
    example_integration_with_classifier()
    example_error_handling()
    example_performance_characteristics()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
