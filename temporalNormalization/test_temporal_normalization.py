"""
Test Suite for Temporal Expression Normalization Module

Comprehensive tests for the temporal normalization system covering:
- Relative expressions
- Weekday expressions
- Time expressions
- Complex combined expressions
- Edge cases and ambiguity handling
"""

import unittest
from datetime import datetime, timedelta
from preprocessing.temporal_normalization import (
    normalize_temporal,
    batch_normalize_temporals,
    TemporalNormalizer,
    TemporalType,
)


class TestTemporalNormalization(unittest.TestCase):
    """Test cases for temporal expression normalization."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Reference date: Tuesday, April 21, 2026, 10:00 AM
        # (Note: April 21, 2026 is actually a Tuesday)
        self.reference_dt = datetime(2026, 4, 21, 10, 0)
        self.normalizer = TemporalNormalizer()
    
    # ========================================================================
    # RELATIVE EXPRESSION TESTS
    # ========================================================================
    
    def test_relative_hoje(self):
        """Test 'hoje' (today)."""
        result = self.normalizer.normalize("hoje", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.RELATIVE)
        self.assertEqual(result.normalized_date, "2026-04-21")
        self.assertTrue(result.confidence >= 0.8)
    
    def test_relative_amanha(self):
        """Test 'amanhã' (tomorrow)."""
        result = self.normalizer.normalize("amanhã", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.RELATIVE)
        self.assertEqual(result.normalized_date, "2026-04-22")
        self.assertTrue(result.confidence >= 0.8)
    
    def test_relative_ontem(self):
        """Test 'ontem' (yesterday)."""
        result = self.normalizer.normalize("ontem", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.RELATIVE)
        self.assertEqual(result.normalized_date, "2026-04-20")
        self.assertTrue(result.confidence >= 0.8)
    
    def test_relative_depois_de_amanha(self):
        """Test 'depois de amanhã' (day after tomorrow)."""
        result = self.normalizer.normalize("depois de amanhã", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.RELATIVE)
        self.assertEqual(result.normalized_date, "2026-04-23")
    
    def test_relative_this_week(self):
        """Test 'esta semana' (this week)."""
        result = self.normalizer.normalize("esta semana", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.INTERVAL)
        self.assertIsNotNone(result.interval_start)
        self.assertIsNotNone(result.interval_end)
    
    def test_relative_next_week(self):
        """Test 'para a semana' (next week)."""
        result = self.normalizer.normalize("para a semana", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.INTERVAL)
        self.assertIsNotNone(result.interval_start)
        self.assertIsNotNone(result.interval_end)
    
    # ========================================================================
    # WEEKDAY TESTS
    # ========================================================================
    
    def test_weekday_segunda(self):
        """Test 'segunda' (Monday) - next occurrence."""
        # Reference is Tuesday, so "segunda" should refer to next Monday (6 days away)
        result = self.normalizer.normalize("segunda", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.DATE)
        # Next Monday is 6 days away from Tuesday
        expected_date = self.reference_dt + timedelta(days=6)
        self.assertEqual(result.normalized_date, expected_date.strftime('%Y-%m-%d'))
    
    def test_weekday_sexta(self):
        """Test 'sexta' (Friday) - within this week."""
        # Reference is Tuesday, so Friday is 3 days ahead
        result = self.normalizer.normalize("sexta", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.DATE)
        expected_date = self.reference_dt + timedelta(days=3)
        self.assertEqual(result.normalized_date, expected_date.strftime('%Y-%m-%d'))
    
    def test_weekday_proxima_terca(self):
        """Test 'próxima terça' (next Tuesday)."""
        result = self.normalizer.normalize("próxima terça", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.DATE)
        # Next Tuesday from Tuesday is 7 days
        expected_date = self.reference_dt + timedelta(days=7)
        self.assertEqual(result.normalized_date, expected_date.strftime('%Y-%m-%d'))
    
    def test_weekday_with_full_name(self):
        """Test 'segunda-feira' (full weekday name)."""
        result = self.normalizer.normalize("segunda-feira", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.DATE)
        self.assertIsNotNone(result.normalized_date)
    
    def test_weekday_ambiguity_same_day_with_past_time(self):
        """Test ambiguity: different weekday with past time."""
        # Reference: Tuesday 10:00
        # Expression: "segunda às 09h" (Monday at 9 AM)
        # Should give next Monday (6 days from Tuesday = April 27)
        result = self.normalizer.normalize("segunda às 09h", self.reference_dt)
        
        expected = self.reference_dt + timedelta(days=6)
        expected = expected.replace(hour=9, minute=0)
        self.assertEqual(result.normalized_datetime, expected)
    
    def test_weekday_ambiguity_same_day_with_future_time(self):
        """Test ambiguity: same weekday with future time."""
        # Reference: Monday 10:00
        # Expression: "segunda às 15h" (Monday at 3 PM - later today)
        result = self.normalizer.normalize("segunda às 15h", self.reference_dt)
        
        expected = self.reference_dt.replace(hour=15, minute=0)
        self.assertEqual(result.normalized_datetime, expected)
    
    # ========================================================================
    # TIME EXPRESSION TESTS
    # ========================================================================
    
    def test_time_exact_hour(self):
        """Test exact hour (e.g., '15h')."""
        result = self.normalizer.normalize("15h", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.TIME)
        self.assertEqual(result.normalized_time, "15:00:00")
    
    def test_time_hour_and_minutes(self):
        """Test hour and minutes (e.g., '14:30')."""
        result = self.normalizer.normalize("14:30", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.TIME)
        self.assertEqual(result.normalized_time, "14:30:00")
    
    def test_time_hour_and_minutes_with_prefix(self):
        """Test time with 'às' prefix (e.g., 'às 15h')."""
        result = self.normalizer.normalize("às 15h", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.TIME)
        self.assertEqual(result.normalized_time, "15:00:00")
    
    def test_time_hour_and_minutes_dot_format(self):
        """Test time with dot format (e.g., '15.30')."""
        result = self.normalizer.normalize("15.30", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.TIME)
        self.assertEqual(result.normalized_time, "15:30:00")
    
    def test_time_with_h_and_minutes(self):
        """Test time format 'HhMM' (e.g., '15h30')."""
        result = self.normalizer.normalize("15h30", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.TIME)
        self.assertEqual(result.normalized_time, "15:30:00")
    
    # ========================================================================
    # TIME OF DAY TESTS
    # ========================================================================
    
    def test_time_of_day_manha(self):
        """Test 'manhã' (morning)."""
        result = self.normalizer.normalize("manhã", self.reference_dt)
        
        # Should be treated as interval (morning hours)
        self.assertIsNotNone(result.normalized_datetime)
    
    def test_time_of_day_tarde(self):
        """Test 'à tarde' (afternoon)."""
        result = self.normalizer.normalize("à tarde", self.reference_dt)
        
        self.assertIsNotNone(result.normalized_datetime)
    
    def test_time_of_day_noite(self):
        """Test 'de noite' (evening/night)."""
        result = self.normalizer.normalize("de noite", self.reference_dt)
        
        self.assertIsNotNone(result.normalized_datetime)
    
    def test_time_of_day_depois_almoço(self):
        """Test 'depois de almoço' (after lunch)."""
        result = self.normalizer.normalize("depois de almoço", self.reference_dt)
        
        self.assertIsNotNone(result.normalized_datetime)
        # Should be in early afternoon (around 13:00-14:00)
        self.assertGreaterEqual(result.normalized_datetime.hour, 12)
        self.assertLessEqual(result.normalized_datetime.hour, 15)
    
    # ========================================================================
    # COMPLEX EXPRESSION TESTS
    # ========================================================================
    
    def test_complex_weekday_with_time(self):
        """Test 'sexta às 15h' (Friday at 3 PM)."""
        result = self.normalizer.normalize("sexta às 15h", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.DATETIME)
        # Friday this week (reference is Tuesday, Friday is 3 days away)
        expected = self.reference_dt + timedelta(days=3)
        expected = expected.replace(hour=15, minute=0)
        self.assertEqual(result.normalized_datetime, expected)
    
    def test_complex_relative_with_time(self):
        """Test 'amanhã às 14h' (tomorrow at 2 PM)."""
        result = self.normalizer.normalize("amanhã às 14h", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.DATETIME)
        expected = self.reference_dt + timedelta(days=1)
        expected = expected.replace(hour=14, minute=0)
        self.assertEqual(result.normalized_datetime, expected)
    
    def test_complex_weekday_with_time_of_day(self):
        """Test 'sexta à tarde' (Friday afternoon)."""
        result = self.normalizer.normalize("sexta à tarde", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.DATETIME)
        # Should be Friday at some afternoon hour (Tuesday + 3 = Friday)
        expected_date = self.reference_dt + timedelta(days=3)
        self.assertEqual(result.normalized_date, expected_date.strftime('%Y-%m-%d'))
        self.assertGreater(result.normalized_datetime.hour, 11)  # Afternoon
    
    def test_complex_explicit_date_with_time(self):
        """Test '16 de Abril às 15h' (April 16 at 3 PM)."""
        result = self.normalizer.normalize("16 de Abril às 15h", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.DATETIME)
        self.assertEqual(result.normalized_date, "2026-04-16")
        self.assertEqual(result.normalized_time, "15:00:00")
    
    # ========================================================================
    # EXPLICIT DATE TESTS
    # ========================================================================
    
    def test_explicit_date_with_month_name(self):
        """Test '16 de Abril' (16 April)."""
        result = self.normalizer.normalize("16 de Abril", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.DATE)
        self.assertEqual(result.normalized_date, "2026-04-16")
    
    def test_explicit_date_with_forward_slash(self):
        """Test '16/04/2026' format."""
        result = self.normalizer.normalize("16/04/2026", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.DATE)
        self.assertEqual(result.normalized_date, "2026-04-16")
    
    def test_explicit_date_with_dash(self):
        """Test '16-04-2026' format."""
        result = self.normalizer.normalize("16-04-2026", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.DATE)
        self.assertEqual(result.normalized_date, "2026-04-16")
    
    def test_explicit_date_without_year(self):
        """Test '16 de Abril' without year (should use reference year)."""
        result = self.normalizer.normalize("16 de Abril", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.DATE)
        self.assertEqual(result.normalized_date, "2026-04-16")
    
    # ========================================================================
    # BATCH PROCESSING TESTS
    # ========================================================================
    
    def test_batch_normalization(self):
        """Test batch processing of multiple expressions."""
        expressions = [
            "amanhã às 15h",
            "sexta",
            "hoje",
            "próxima segunda",
        ]
        
        results = batch_normalize_temporals(expressions, self.reference_dt)
        
        self.assertEqual(len(results), len(expressions))
        for result in results:
            self.assertIsNotNone(result.normalized_datetime or result.interval_start)
    
    # ========================================================================
    # EDGE CASES AND ERROR HANDLING
    # ========================================================================
    
    def test_invalid_expression(self):
        """Test handling of invalid expressions."""
        result = self.normalizer.normalize("xyz123invalid", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.UNKNOWN)
        self.assertEqual(result.confidence, 0.0)
    
    def test_partially_invalid_expression(self):
        """Test expression with some valid parts."""
        result = self.normalizer.normalize("sexta e xyz", self.reference_dt)
        
        # Should still parse "sexta" part
        self.assertNotEqual(result.temporal_type, TemporalType.UNKNOWN)
    
    def test_empty_string(self):
        """Test handling of empty string."""
        result = self.normalizer.normalize("", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.UNKNOWN)
    
    def test_whitespace_only(self):
        """Test handling of whitespace-only string."""
        result = self.normalizer.normalize("   ", self.reference_dt)
        
        self.assertEqual(result.temporal_type, TemporalType.UNKNOWN)
    
    def test_case_insensitivity(self):
        """Test that parsing is case-insensitive."""
        result1 = self.normalizer.normalize("SEGUNDA", self.reference_dt)
        result2 = self.normalizer.normalize("segunda", self.reference_dt)
        result3 = self.normalizer.normalize("Segunda", self.reference_dt)
        
        self.assertEqual(result1.normalized_date, result2.normalized_date)
        self.assertEqual(result2.normalized_date, result3.normalized_date)
    
    def test_extra_whitespace(self):
        """Test handling of extra whitespace."""
        result1 = self.normalizer.normalize("sexta   às   15h", self.reference_dt)
        result2 = self.normalizer.normalize("sexta às 15h", self.reference_dt)
        
        self.assertEqual(result1.normalized_datetime, result2.normalized_datetime)
    
    # ========================================================================
    # DEFAULT REFERENCE DATETIME
    # ========================================================================
    
    def test_default_reference_datetime(self):
        """Test that None reference datetime defaults to now."""
        result = self.normalizer.normalize("hoje")
        
        self.assertEqual(result.temporal_type, TemporalType.RELATIVE)
        # Should be today's date (approximately now)
        self.assertEqual(
            result.normalized_date,
            datetime.now().strftime('%Y-%m-%d')
        )


class TestTemporalNormalizationIntegration(unittest.TestCase):
    """Integration tests for real email scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reference_dt = datetime(2026, 4, 21, 10, 0)  # Tuesday, April 21, 2026 at 10 AM
        self.normalizer = TemporalNormalizer()
    
    def test_meeting_scheduling_emails(self):
        """Test normalization in typical meeting scheduling scenarios."""
        
        scenarios = [
            ("Consegues reunir segunda às 14h?", TemporalType.DATETIME),
            ("Podemos agendar para amanhã?", TemporalType.RELATIVE),  # amanhã = tomorrow (no time)
            ("Vamos marcar para sexta à tarde?", TemporalType.DATETIME),  # sexta + tarde = datetime
            ("Pode ser ainda no final desta semana?", TemporalType.DATE),  # final desta semana
            ("Para a semana temos tempo?", TemporalType.INTERVAL),
        ]
        
        for expression, expected_type in scenarios:
            result = self.normalizer.normalize(expression, self.reference_dt)
            self.assertEqual(
                result.temporal_type,
                expected_type,
                f"Failed for: {expression}"
            )
    
    def test_to_dict_serialization(self):
        """Test that results can be serialized to dict."""
        result = self.normalizer.normalize("sexta às 15h", self.reference_dt)
        result_dict = result.to_dict()
        
        # Check key fields are present
        self.assertIn('original_text', result_dict)
        self.assertIn('temporal_type', result_dict)
        self.assertIn('normalized_datetime', result_dict)
        self.assertIn('confidence', result_dict)
        
        # All values should be serializable (no datetime objects in dict)
        self.assertIsInstance(result_dict['normalized_datetime'], str)


if __name__ == '__main__':
    unittest.main()
