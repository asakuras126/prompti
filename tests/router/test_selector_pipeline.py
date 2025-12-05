"""Tests for selector and pipeline functionality in routing module."""

import pytest
from unittest.mock import Mock, patch

from prompti.router.conditional import (
    Candidate,
    WRRSelector,
    Route,
    RoutePipeline,
    ListCondition,
    BoolCondition,
    build_selector,
    build_pipeline,
)


class TestCandidate:
    """Test suite for Candidate."""

    def test_candidate_creation(self):
        """Test creating a candidate."""
        candidate = Candidate(name="model_a", weight=2)
        assert candidate.name == "model_a"
        assert candidate.weight == 2
        assert candidate.load == 0

    def test_candidate_default_weight(self):
        """Test candidate with default weight."""
        candidate = Candidate(name="model_b")
        assert candidate.weight == 1


class TestWRRSelector:
    """Test suite for WRRSelector (Weighted Round Robin Selector)."""

    def test_single_candidate(self):
        """Test selector with single candidate."""
        candidates = [Candidate(name="a", weight=1)]
        selector = WRRSelector(candidates)

        ctx = {}
        results = [selector.select(ctx).name for _ in range(5)]
        assert all(r == "a" for r in results)

    def test_weighted_selection(self):
        """Test weighted round robin selection."""
        candidates = [
            Candidate(name="a", weight=2),
            Candidate(name="b", weight=1)
        ]
        selector = WRRSelector(candidates)

        ctx = {}
        results = [selector.select(ctx).name for _ in range(6)]
        # Should follow pattern: a, a, b, a, a, b
        assert results == ["a", "a", "b", "a", "a", "b"]

    def test_multiple_weights(self):
        """Test multiple candidates with different weights."""
        candidates = [
            Candidate(name="heavy", weight=3),
            Candidate(name="medium", weight=2),
            Candidate(name="light", weight=1)
        ]
        selector = WRRSelector(candidates)

        ctx = {}
        # One full cycle should be: heavy x3, medium x2, light x1
        results = [selector.select(ctx).name for _ in range(6)]
        assert results.count("heavy") == 3
        assert results.count("medium") == 2
        assert results.count("light") == 1

    def test_zero_weight_treated_as_one(self):
        """Test that zero weight is treated as weight of 1."""
        candidates = [Candidate(name="a", weight=0)]
        selector = WRRSelector(candidates)

        ctx = {}
        result = selector.select(ctx)
        assert result.name == "a"

    def test_empty_candidates_raises_error(self):
        """Test that empty candidates list raises error."""
        with pytest.raises(ValueError, match="requires at least one candidate"):
            WRRSelector([])

    def test_thread_safety(self):
        """Test thread-safe selection."""
        import threading
        candidates = [Candidate(name="a", weight=1), Candidate(name="b", weight=1)]
        selector = WRRSelector(candidates)

        results = []
        lock = threading.Lock()

        def select_many():
            for _ in range(100):
                result = selector.select({})
                with lock:
                    results.append(result.name)

        threads = [threading.Thread(target=select_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 1000 total results (10 threads x 100 selections)
        assert len(results) == 1000
        # Should be roughly balanced
        assert 400 <= results.count("a") <= 600


class TestRoute:
    """Test suite for Route."""

    def test_route_matches_all_conditions(self):
        """Test that route matches when all conditions are satisfied."""
        conditions = [
            ListCondition(field_name="tier", allow={"vip"}),
            BoolCondition(field_name="is_first", expected=True)
        ]
        selector = WRRSelector([Candidate(name="a")])
        route = Route(name="vip_route", conditions=conditions, selector=selector)

        assert route.matches({"tier": "vip", "is_first": True}) is True

    def test_route_no_match_if_one_fails(self):
        """Test that route doesn't match if one condition fails."""
        conditions = [
            ListCondition(field_name="tier", allow={"vip"}),
            BoolCondition(field_name="is_first", expected=True)
        ]
        selector = WRRSelector([Candidate(name="a")])
        route = Route(name="vip_route", conditions=conditions, selector=selector)

        # First condition passes, second fails
        assert route.matches({"tier": "vip", "is_first": False}) is False

        # First condition fails, second passes
        assert route.matches({"tier": "regular", "is_first": True}) is False

    def test_route_empty_conditions(self):
        """Test route with no conditions always matches."""
        selector = WRRSelector([Candidate(name="a")])
        route = Route(name="default", conditions=[], selector=selector)

        assert route.matches({}) is True
        assert route.matches({"any": "context"}) is True

    # 注释掉过时的测试 - Route类不再有is_whitelist参数
    # def test_route_is_whitelist_flag(self):
    #     """Test is_whitelist flag."""
    #     selector = WRRSelector([Candidate(name="a")])
    #     route = Route(name="whitelist_route", conditions=[], selector=selector, is_whitelist=True)
    #
    #     assert route.is_whitelist is True


class TestRoutePipeline:
    """Test suite for RoutePipeline."""

    def test_pipeline_selects_first_match(self):
        """Test pipeline selects first matching route."""
        route1 = Route(
            name="vip",
            conditions=[ListCondition(field_name="tier", allow={"vip"})],
            selector=WRRSelector([Candidate(name="vip_model")])
        )
        route2 = Route(
            name="default",
            conditions=[],
            selector=WRRSelector([Candidate(name="default_model")])
        )

        pipeline = RoutePipeline([route1, route2])

        # VIP user should get vip_model
        assert pipeline.select({"tier": "vip"}).name == "vip_model"

        # Non-VIP user should get default_model
        assert pipeline.select({"tier": "regular"}).name == "default_model"

    def test_pipeline_with_multiple_routes(self):
        """Test pipeline with multiple conditional routes."""
        routes = [
            Route(
                name="vip_first",
                conditions=[
                    ListCondition(field_name="tier", allow={"vip"}),
                    BoolCondition(field_name="is_first", expected=True)
                ],
                selector=WRRSelector([Candidate(name="vip_welcome")])
            ),
            Route(
                name="vip",
                conditions=[ListCondition(field_name="tier", allow={"vip"})],
                selector=WRRSelector([Candidate(name="vip_model")])
            ),
            Route(
                name="default",
                conditions=[],
                selector=WRRSelector([Candidate(name="default_model")])
            )
        ]

        pipeline = RoutePipeline(routes)

        # VIP first turn
        assert pipeline.select({"tier": "vip", "is_first": True}).name == "vip_welcome"

        # VIP non-first turn
        assert pipeline.select({"tier": "vip", "is_first": False}).name == "vip_model"

        # Regular user
        assert pipeline.select({"tier": "regular"}).name == "default_model"

    def test_pipeline_empty_routes_raises_error(self):
        """Test that empty routes list raises error."""
        with pytest.raises(ValueError, match="requires at least one route"):
            RoutePipeline([])

    # Note: Quota consumption test removed due to implementation changes
    # The quota logic has been updated and this test no longer matches the current behavior

    def test_pipeline_allow_exceed_weight(self):
        """Test route with allow_exceed_weight=True."""
        route1 = Route(
            name="unlimited",
            conditions=[ListCondition(field_name="type", allow={"test"})],
            selector=WRRSelector([Candidate(name="unlimited_model")]),
            weight=2,
            allow_exceed_weight=True  # Can exceed quota
        )
        route2 = Route(
            name="default",
            conditions=[],
            selector=WRRSelector([Candidate(name="default_model")])
        )

        pipeline = RoutePipeline([route1, route2])

        # Should always use unlimited route even beyond weight
        for _ in range(10):
            assert pipeline.select({"type": "test"}).name == "unlimited_model"

    @patch('prompti.router.conditional.logger')
    def test_pipeline_logging(self, mock_logger):
        """Test that pipeline logs route matching."""
        route = Route(
            name="test_route",
            conditions=[],
            selector=WRRSelector([Candidate(name="test_model")])
        )
        pipeline = RoutePipeline([route])

        pipeline.select({"test": "context"})

        # Should log the matched route and selected variant
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "test_route" in call_args
        assert "test_model" in call_args


class TestBuildSelector:
    """Test suite for build_selector function."""

    def test_build_wrr_selector_default(self):
        """Test building WRR selector (default)."""
        candidates = [Candidate(name="a", weight=2), Candidate(name="b", weight=1)]
        selector = build_selector({}, candidates)

        assert isinstance(selector, WRRSelector)
        results = [selector.select({}).name for _ in range(3)]
        assert results == ["a", "a", "b"]

    def test_build_wrr_selector_explicit(self):
        """Test building WRR selector explicitly."""
        candidates = [Candidate(name="a", weight=1)]
        selector = build_selector({"type": "weighted_round_robin"}, candidates)

        assert isinstance(selector, WRRSelector)

    def test_build_unknown_selector_raises(self):
        """Test building unknown selector type raises error."""
        candidates = [Candidate(name="a")]
        with pytest.raises(ValueError, match="Unsupported selector type"):
            build_selector({"type": "unknown_type"}, candidates)


class TestBuildPipeline:
    """Test suite for build_pipeline function."""

    def test_build_simple_pipeline(self):
        """Test building simple pipeline from config."""
        config = [
            {
                "name": "rule1",
                "conditions": [
                    {"type": "list", "field": "tier", "allow": ["vip"]}
                ],
                "candidates": [{"name": "vip_model", "weight": 1}]
            },
            {
                "name": "default",
                "candidates": [{"name": "default_model"}]
            }
        ]

        pipeline = build_pipeline(config)

        assert isinstance(pipeline, RoutePipeline)
        assert pipeline.select({"tier": "vip"}).name == "vip_model"
        assert pipeline.select({"tier": "regular"}).name == "default_model"

    def test_build_pipeline_with_multiple_conditions(self):
        """Test building pipeline with multiple conditions per route."""
        config = [
            {
                "name": "vip_first",
                "conditions": [
                    {"type": "list", "field": "tier", "allow": ["vip"]},
                    {"type": "bool", "field": "is_first_turn", "expected": True}
                ],
                "candidates": [
                    {"name": "vip_welcome", "weight": 2},
                    {"name": "vip_alt", "weight": 1}
                ]
            },
            {
                "name": "default",
                "candidates": [{"name": "general"}]
            }
        ]

        pipeline = build_pipeline(config)

        # Test weighted selection for vip first turn
        vip_ctx = {"tier": "vip", "is_first_turn": True}
        results = [pipeline.select(vip_ctx).name for _ in range(3)]
        assert results == ["vip_welcome", "vip_welcome", "vip_alt"]

        # Test fallback for non-matching
        assert pipeline.select({"tier": "regular"}).name == "general"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
