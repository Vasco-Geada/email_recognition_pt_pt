"""
TEMPORAL NORMALIZATION: USAGE EXAMPLES & QUICK START

Practical introduction to using the temporal normalization module.
"""

from datetime import datetime, timedelta
from preprocessing.temporal_normalization import (
    normalize_temporal,
    batch_normalize_temporals,
    TemporalNormalizer,
    TemporalType,
)
import json


# ==============================================================================
# QUICK START
# ==============================================================================

def example_1_basic_usage():
    """Most basic usage: normalize a single expression."""
    
    # Define a reference date (when the email was received)
    reference_date = datetime(2026, 4, 21, 10, 0)  # Monday, April 21, 2026 at 10 AM
    
    # Normalize an expression
    result = normalize_temporal(
        temporal_expression="sexta às 15h",
        reference_datetime=reference_date
    )
    
    # Access results
    print("Original expression:", result.original_text)
    print("Type:", result.temporal_type.value)
    print("Normalized datetime:", result.normalized_datetime_str)
    print("Confidence:", result.confidence)
    
    # Output:
    # Original expression: sexta às 15h
    # Type: datetime
    # Normalized datetime: 2026-04-24T15:00:00
    # Confidence: 0.9


def example_2_batch_processing():
    """Normalize multiple expressions from a single email."""
    
    reference_date = datetime(2026, 4, 21, 10, 0)
    
    # Email might contain multiple temporal references
    email_body = """
    Olá,
    
    Consegues reunir amanhã às 14h? Se não conseguir, posso fazer sexta à tarde.
    Também poderia ser no final desta semana.
    
    Obrigado
    """
    
    # Temporal expressions already extracted (by extraction module)
    temporal_expressions = [
        "amanhã às 14h",
        "sexta à tarde",
        "final desta semana",
    ]
    
    # Batch normalize
    results = batch_normalize_temporals(temporal_expressions, reference_date)
    
    for expr, result in zip(temporal_expressions, results):
        print(f"Expression: '{expr}'")
        print(f"  → Date: {result.normalized_date}")
        print(f"  → Time: {result.normalized_time}")
        print(f"  → Type: {result.temporal_type.value}")
        print()


def example_3_error_handling():
    """Handle parsing errors gracefully."""
    
    reference_date = datetime(2026, 4, 21, 10, 0)
    
    expressions = [
        "sexta às 15h",        # ✓ Valid
        "xyz123",              # ✗ Invalid
        "amanhã",              # ✓ Valid
        "",                    # ✗ Empty
        "segunda",             # ✓ Valid
    ]
    
    results = batch_normalize_temporals(expressions, reference_date)
    
    for expr, result in zip(expressions, results):
        if result.temporal_type == TemporalType.UNKNOWN:
            print(f"⚠️  Could not parse: '{expr}'")
            if result.confidence < 0.5:
                print(f"   Reason: {result.notes}")
        else:
            print(f"✓ '{expr}' → {result.normalized_datetime_str}")


def example_4_confidence_scores():
    """Understand confidence scores."""
    
    reference_date = datetime(2026, 4, 21, 10, 0)
    
    expressions = [
        ("16 de Abril de 2026 às 15h", "Explicit date + time → HIGH confidence"),
        ("sexta às 15h", "Weekday + exact time → HIGH confidence"),
        ("amanhã", "Relative expression → MEDIUM-HIGH confidence"),
        ("15h", "Time only → MEDIUM confidence"),
        ("sexta à tarde", "Weekday + approximate time → MEDIUM confidence"),
        ("em breve", "Vague expression → LOW confidence"),
    ]
    
    for expr, description in expressions:
        result = normalize_temporal(expr, reference_date)
        print(f"'{expr}'")
        print(f"  {description}")
        print(f"  Confidence: {result.confidence} | Type: {result.temporal_type.value}")
        print()


def example_5_temporal_types():
    """Different types of normalized temporal expressions."""
    
    reference_date = datetime(2026, 4, 21, 10, 0)
    
    test_cases = {
        TemporalType.DATE: ["sexta", "segunda-feira"],
        TemporalType.TIME: ["15h", "às 14:30"],
        TemporalType.DATETIME: ["sexta às 15h", "amanhã às 14h"],
        TemporalType.INTERVAL: ["para a semana", "este mês"],
        TemporalType.RELATIVE: ["amanhã", "ontem"],
    }
    
    for temporal_type, expressions in test_cases.items():
        for expr in expressions:
            result = normalize_temporal(expr, reference_date)
            print(f"[{result.temporal_type.value.upper()}] {expr}")
            if result.temporal_type == TemporalType.INTERVAL:
                print(f"  Range: {result.interval_start_str} to {result.interval_end_str}")
            else:
                print(f"  DateTime: {result.normalized_datetime_str}")
        print()


def example_6_ambiguity_handling():
    """How the system handles ambiguous expressions."""
    
    # Same weekday without explicit time
    reference_date = datetime(2026, 4, 21, 10, 0)  # Monday 10 AM
    
    # Case 1: Past time on same weekday
    result1 = normalize_temporal("segunda às 09h", reference_date)
    print("Case 1: 'segunda às 09h' at Monday 10 AM")
    print(f"  Result: {result1.normalized_datetime_str}")
    print(f"  Interpretation: NEXT Monday (because 09h < 10h)")
    print(f"  Note: {result1.notes}")
    print()
    
    # Case 2: Future time on same weekday
    result2 = normalize_temporal("segunda às 15h", reference_date)
    print("Case 2: 'segunda às 15h' at Monday 10 AM")
    print(f"  Result: {result2.normalized_datetime_str}")
    print(f"  Interpretation: TODAY (because 15h > 10h)")
    print()
    
    # Case 3: No time specification (ambiguous)
    result3 = normalize_temporal("segunda", reference_date)
    print("Case 3: 'segunda' (no time) at Monday 10 AM")
    print(f"  Result: {result3.normalized_datetime_str}")
    print(f"  Interpretation: NEXT Monday (conservative default)")
    print(f"  Note: {result3.notes}")


def example_7_context_usage():
    """Using additional context for better parsing."""
    
    reference_date = datetime(2026, 4, 21, 10, 0)
    normalizer = TemporalNormalizer()
    
    # Optional context can be added (email intent, previous expressions, etc.)
    context = {
        'email_intent': 'agendamento_reuniao',  # Scheduling
        'previous_temporal': None,               # For discourse resolution
    }
    
    # Current implementation doesn't use context heavily,
    # but infrastructure is in place for future enhancements
    result = normalizer.normalize(
        "sexta",
        reference_datetime=reference_date,
        context=context
    )
    
    print(f"Expression: 'sexta'")
    print(f"Context: {context}")
    print(f"Result: {result.normalized_datetime_str}")


def example_8_integration_with_pipeline():
    """Integration with email processing pipeline."""
    
    # Simulated email document
    email_dict = {
        'subject': 'Reunião marcada para sexta',
        'body': 'Consegues reunir sexta às 15h? Se não conseguir, segunda de manhã?',
        'received_date': datetime(2026, 4, 21, 10, 0),
        'sender': 'joao@example.com',
    }
    
    # Step 1: Extract raw temporal expressions (from another module)
    temporal_expressions = [
        "sexta às 15h",
        "segunda de manhã",
    ]
    
    # Step 2: Normalize each expression
    normalizer = TemporalNormalizer()
    normalized_temporals = []
    
    for expr in temporal_expressions:
        normalized = normalizer.normalize(
            expr,
            reference_datetime=email_dict['received_date']
        )
        normalized_temporals.append(normalized.to_dict())
    
    # Step 3: Results can be stored or processed further
    results = {
        'email_id': email_dict.get('id'),
        'received_date': email_dict['received_date'].isoformat(),
        'temporal_expressions': normalized_temporals,
    }
    
    print("Integration Result:")
    print(json.dumps(results, indent=2, default=str))


def example_9_precision_levels():
    """Different precision levels in normalization."""
    
    reference_date = datetime(2026, 4, 21, 10, 0)
    normalizer = TemporalNormalizer()
    
    expressions = [
        ("16 de Abril de 2026 às 15h", "Exact"),
        ("sexta às 15h", "Exact"),
        ("sexta à tarde", "Approximate"),
        ("para a semana", "Vague (week-level)"),
        ("em breve", "Very Vague"),
    ]
    
    for expr, expected_precision in expressions:
        result = normalizer.normalize(expr, reference_date)
        print(f"'{expr}'")
        print(f"  Expected precision: {expected_precision}")
        print(f"  Actual precision: {result.precision}")
        print(f"  DateTime: {result.normalized_datetime_str}")
        if result.interval_start_str:
            print(f"  Interval: [{result.interval_start_str}, {result.interval_end_str}]")
        print()


def example_10_advanced_complex_expressions():
    """Handling complex multi-part expressions."""
    
    reference_date = datetime(2026, 4, 21, 10, 0)
    normalizer = TemporalNormalizer()
    
    # These are harder to parse
    complex_expressions = [
        "sexta à tarde, perto das 15h",
        "próxima segunda de manhã às 10h",
        "amanhã depois de almoço",
        "final desta semana, sábado ou domingo",
    ]
    
    for expr in complex_expressions:
        result = normalizer.normalize(expr, reference_date)
        print(f"Complex: '{expr}'")
        print(f"  Type: {result.temporal_type.value}")
        print(f"  DateTime: {result.normalized_datetime_str}")
        print(f"  Confidence: {result.confidence}")
        if result.notes:
            print(f"  Notes: {'; '.join(result.notes)}")
        print()


# ==============================================================================
# REAL-WORLD USE CASE: EMAIL PROCESSING
# ==============================================================================

def example_real_world_email_processing():
    """Complete integration example with a real email."""
    
    # Sample email (meeting scheduling in Portuguese)
    email = {
        'id': 'msg_001',
        'received_date': datetime(2026, 4, 21, 10, 30),
        'subject': 'Re: Reunião de projeto',
        'body': '''
        Olá Ana,
        
        Obrigado pelo email. Conseguimos reunir sexta à tarde? 
        Prefiro a partir das 14h.
        
        Se não conseguir, posso fazer segunda de manhã.
        
        Confirmação rápida pls
        
        Regards,
        João
        '''
    }
    
    # Initialize normalizer
    reference_datetime = email['received_date']
    normalizer = TemporalNormalizer()
    
    # Simulate extraction of temporal expressions (done by separate module)
    extracted_temporals = [
        "sexta à tarde",
        "a partir das 14h",
        "segunda de manhã",
    ]
    
    print("=" * 70)
    print("EMAIL TEMPORAL PROCESSING")
    print("=" * 70)
    print(f"Email Subject: {email['subject']}")
    print(f"Received: {email['received_date'].isoformat()}")
    print("-" * 70)
    
    # Process each temporal expression
    meeting_options = []
    for temporal_expr in extracted_temporals:
        result = normalizer.normalize(temporal_expr, reference_datetime)
        
        if result.temporal_type != TemporalType.UNKNOWN:
            meeting_options.append({
                'original': temporal_expr,
                'normalized_datetime': result.normalized_datetime_str,
                'type': result.temporal_type.value,
                'precision': result.precision,
                'confidence': result.confidence,
            })
            
            print(f"\n✓ Temporal Expression: '{temporal_expr}'")
            print(f"  → Normalized: {result.normalized_datetime_str}")
            print(f"  → Type: {result.temporal_type.value}")
            print(f"  → Confidence: {result.confidence:.0%}")
        else:
            print(f"\n✗ Could not parse: '{temporal_expr}'")
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(meeting_options)} meeting options extracted")
    print("=" * 70)
    
    # Create structured output
    structured_output = {
        'email_id': email['id'],
        'received_date': email['received_date'].isoformat(),
        'meeting_options': meeting_options,
    }
    
    print("\nStructured Output (JSON):")
    print(json.dumps(structured_output, indent=2))
    
    return structured_output


# ==============================================================================
# TESTING & VALIDATION
# ==============================================================================

def example_testing():
    """Create test cases for validation."""
    
    reference_date = datetime(2026, 4, 21, 10, 0)
    
    test_cases = [
        # (expression, expected_date, expected_time)
        ("sexta", "2026-04-24", None),
        ("segunda", "2026-04-28", None),  # Next Monday
        ("amanhã", "2026-04-22", None),
        ("hoje", "2026-04-21", None),
        ("sexta às 15h", "2026-04-24", "15:00:00"),
        ("16 de Abril", "2026-04-16", None),
        ("16 de Abril às 14h", "2026-04-16", "14:00:00"),
    ]
    
    normalizer = TemporalNormalizer()
    passed = 0
    failed = 0
    
    print("Running test cases...")
    print("-" * 70)
    
    for expr, expected_date, expected_time in test_cases:
        result = normalizer.normalize(expr, reference_date)
        
        # Validate
        date_ok = (result.normalized_date == expected_date)
        time_ok = (expected_time is None or result.normalized_time == expected_time)
        
        if date_ok and time_ok:
            print(f"✓ PASS: '{expr}'")
            passed += 1
        else:
            print(f"✗ FAIL: '{expr}'")
            print(f"    Expected: {expected_date} {expected_time or ''}")
            print(f"    Got:      {result.normalized_date} {result.normalized_time or ''}")
            failed += 1
    
    print("-" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)}")
    
    return passed, failed


# ==============================================================================
# PERFORMANCE & METRICS
# ==============================================================================

def example_performance_monitoring():
    """Monitor performance metrics."""
    
    import time
    
    reference_date = datetime(2026, 4, 21, 10, 0)
    expressions = [
        "sexta às 15h",
        "amanhã",
        "segunda-feira",
        "16 de Abril",
        "próxima terça",
    ] * 100  # Test on 500 expressions
    
    normalizer = TemporalNormalizer()
    
    # Measure performance
    latencies = []
    successes = 0
    
    start_time = time.time()
    for expr in expressions:
        expr_start = time.time()
        result = normalizer.normalize(expr, reference_date)
        expr_latency = (time.time() - expr_start) * 1000  # Convert to ms
        latencies.append(expr_latency)
        
        if result.temporal_type != TemporalType.UNKNOWN:
            successes += 1
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    min_latency = min(latencies)
    
    print("PERFORMANCE METRICS")
    print("=" * 70)
    print(f"Total expressions: {len(expressions)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {len(expressions) / total_time:.0f} expr/sec")
    print(f"Success rate: {successes / len(expressions):.1%}")
    print(f"Average latency: {avg_latency:.2f}ms")
    print(f"Min/Max latency: {min_latency:.2f}ms / {max_latency:.2f}ms")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("TEMPORAL NORMALIZATION - EXAMPLES & QUICK START")
    print("=" * 70 + "\n")
    
    # Run examples
    examples = [
        ("1. Basic Usage", example_1_basic_usage),
        ("2. Batch Processing", example_2_batch_processing),
        ("3. Error Handling", example_3_error_handling),
        ("4. Confidence Scores", example_4_confidence_scores),
        ("5. Temporal Types", example_5_temporal_types),
        ("6. Ambiguity Handling", example_6_ambiguity_handling),
        ("7. Context Usage", example_7_context_usage),
        ("8. Pipeline Integration", example_8_integration_with_pipeline),
        ("9. Precision Levels", example_9_precision_levels),
        ("10. Complex Expressions", example_10_advanced_complex_expressions),
        ("Real World Example", example_real_world_email_processing),
        ("Testing", example_testing),
        ("Performance Monitoring", example_performance_monitoring),
    ]
    
    for name, func in examples:
        print(f"\n{'=' * 70}")
        print(f"{name}")
        print('=' * 70)
        try:
            func()
        except Exception as e:
            print(f"Error running {name}: {e}")
            import traceback
            traceback.print_exc()
        print()
