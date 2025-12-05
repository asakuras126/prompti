"""
Test variant routing metrics reporting functionality.

This test verifies that variant routing decisions from conditional router pipeline
are correctly reported to Prometheus.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import pathlib

# Add src to path
test_dir = pathlib.Path(__file__).resolve().parents[1]
src_path = test_dir / 'src'
sys.path.insert(0, str(src_path))

from prompti.router.conditional import build_pipeline
from prompti.otel import LLMMetrics


class VariantRoutingMetricsTests(unittest.TestCase):
    """Test variant routing metrics reporting."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock LLM metrics instance
        self.mock_metrics = Mock(spec=LLMMetrics)
        self.mock_metrics.enabled = True
        self.mock_metrics.record_variant_routing = Mock()

    def test_normal_routing_metrics(self):
        """Test that normal routing decisions are reported with routing_source='normal'."""
        pipeline = build_pipeline([
            {
                'name': 'hash_rule',
                'conditions': [
                    {'type': 'hash', 'hash_key': 'session_id', 'percentage': 50.0}
                ],
                'candidates': [
                    {'name': 'variant_a', 'weight': 1}
                ]
            },
            {
                'name': 'default',
                'candidates': [
                    {'name': 'variant_b', 'weight': 1}
                ]
            }
        ])

        with patch('prompti.otel.get_llm_metrics', return_value=self.mock_metrics):
            # Test hash match
            ctx = {
                'session_id': 'test_session_123',
                'template_name': 'test_template'
            }
            candidate = pipeline.select(ctx)

            # Verify metrics were recorded
            self.mock_metrics.record_variant_routing.assert_called_once()
            call_args = self.mock_metrics.record_variant_routing.call_args
            self.assertEqual(call_args[1]['template_name'], 'test_template')
            self.assertEqual(call_args[1]['variant_name'], candidate.name)
            self.assertEqual(call_args[1]['routing_source'], 'normal')

    def test_all_routes_reported_as_normal(self):
        """Test that all routing decisions through pipeline are reported as normal."""
        pipeline = build_pipeline([
            {
                'name': 'tier_rule',
                'conditions': [
                    {'type': 'value', 'field': 'tier', 'values': ['premium']},
                    {'type': 'value', 'field': 'query_type', 'values': ['coding']}
                ],
                'candidates': [
                    {'name': 'variant_premium', 'weight': 1}
                ]
            },
            {
                'name': 'default',
                'candidates': [
                    {'name': 'variant_default', 'weight': 1}
                ]
            }
        ])

        with patch('prompti.otel.get_llm_metrics', return_value=self.mock_metrics):
            # Test route matching
            ctx = {
                'tier': 'premium',
                'query_type': 'coding',
                'template_name': 'test_template'
            }
            candidate = pipeline.select(ctx)

            # Verify metrics were recorded with normal source
            self.mock_metrics.record_variant_routing.assert_called_once()
            call_args = self.mock_metrics.record_variant_routing.call_args
            self.assertEqual(call_args[1]['template_name'], 'test_template')
            self.assertEqual(call_args[1]['variant_name'], 'variant_premium')
            self.assertEqual(call_args[1]['routing_source'], 'normal')


if __name__ == '__main__':
    unittest.main()
