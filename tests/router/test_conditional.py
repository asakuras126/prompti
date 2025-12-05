"""Tests for conditional routing module."""

import pytest

from prompti.router.conditional import (
    ListCondition,
    BoolCondition,
    ValueCondition,
    MinMaxCondition,
    HashCondition,
)


class TestListCondition:
    """Test suite for ListCondition."""

    def test_allow_list_match(self):
        """Test that values in allow list match."""
        condition = ListCondition(field_name="user_id", allow={"user1", "user2"})
        assert condition.matches({"user_id": "user1"}) is True
        assert condition.matches({"user_id": "USER1"}) is True  # Case insensitive

    def test_allow_list_no_match(self):
        """Test that values not in allow list don't match."""
        condition = ListCondition(field_name="user_id", allow={"user1", "user2"})
        assert condition.matches({"user_id": "user3"}) is False

    def test_deny_list_match(self):
        """Test that values in deny list don't match."""
        condition = ListCondition(field_name="user_id", deny={"blocked1", "blocked2"})
        assert condition.matches({"user_id": "blocked1"}) is False
        assert condition.matches({"user_id": "BLOCKED1"}) is False  # Case insensitive

    def test_deny_list_no_match(self):
        """Test that values not in deny list match."""
        condition = ListCondition(field_name="user_id", deny={"blocked1"})
        assert condition.matches({"user_id": "user1"}) is True

    def test_both_allow_and_deny(self):
        """Test combination of allow and deny lists."""
        condition = ListCondition(
            field_name="user_id",
            allow={"user1", "user2", "user3"},
            deny={"user3"}
        )
        assert condition.matches({"user_id": "user1"}) is True
        assert condition.matches({"user_id": "user3"}) is False  # In deny list
        assert condition.matches({"user_id": "user4"}) is False  # Not in allow list

    def test_missing_field(self):
        """Test behavior with missing field."""
        condition = ListCondition(field_name="user_id", allow={"user1"})
        assert condition.matches({}) is False
        assert condition.matches({"other_field": "value"}) is False

    def test_empty_lists(self):
        """Test with empty allow/deny lists."""
        condition = ListCondition(field_name="user_id")
        assert condition.matches({"user_id": "anyone"}) is True

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        condition = ListCondition(field_name="user_id", allow={"user1"})
        assert condition.matches({"user_id": "USER1"}) is True
        assert condition.matches({"user_id": "User1"}) is True


class TestBoolCondition:
    """Test suite for BoolCondition."""

    def test_true_expected_true_value(self):
        """Test matching when expecting True and value is True."""
        condition = BoolCondition(field_name="is_enabled", expected=True)
        assert condition.matches({"is_enabled": True}) is True
        assert condition.matches({"is_enabled": 1}) is True
        assert condition.matches({"is_enabled": "yes"}) is True

    def test_true_expected_false_value(self):
        """Test not matching when expecting True and value is False."""
        condition = BoolCondition(field_name="is_enabled", expected=True)
        assert condition.matches({"is_enabled": False}) is False
        assert condition.matches({"is_enabled": 0}) is False
        assert condition.matches({"is_enabled": ""}) is False

    def test_false_expected_false_value(self):
        """Test matching when expecting False and value is False."""
        condition = BoolCondition(field_name="is_enabled", expected=False)
        assert condition.matches({"is_enabled": False}) is True
        assert condition.matches({"is_enabled": 0}) is True
        assert condition.matches({"is_enabled": ""}) is True

    def test_false_expected_true_value(self):
        """Test not matching when expecting False and value is True."""
        condition = BoolCondition(field_name="is_enabled", expected=False)
        assert condition.matches({"is_enabled": True}) is False
        assert condition.matches({"is_enabled": 1}) is False

    def test_missing_field(self):
        """Test behavior with missing field."""
        condition = BoolCondition(field_name="is_enabled", expected=False)
        assert condition.matches({}) is True  # Missing field is falsy


class TestValueCondition:
    """Test suite for ValueCondition."""

    def test_value_in_allowed(self):
        """Test matching when value is in allowed set."""
        condition = ValueCondition(field_name="query_type", allowed={"coding", "general"})
        assert condition.matches({"query_type": "coding"}) is True
        assert condition.matches({"query_type": "general"}) is True

    def test_value_not_in_allowed(self):
        """Test not matching when value is not in allowed set."""
        condition = ValueCondition(field_name="query_type", allowed={"coding", "general"})
        assert condition.matches({"query_type": "math"}) is False

    def test_case_insensitive(self):
        """Test case-insensitive matching (default)."""
        condition = ValueCondition(field_name="query_type", allowed={"coding"})
        assert condition.matches({"query_type": "CODING"}) is True
        assert condition.matches({"query_type": "Coding"}) is True

    def test_case_sensitive(self):
        """Test case-sensitive matching."""
        condition = ValueCondition(
            field_name="query_type",
            allowed={"coding"},
            case_sensitive=True
        )
        assert condition.matches({"query_type": "coding"}) is True
        assert condition.matches({"query_type": "CODING"}) is False

    def test_missing_field(self):
        """Test behavior with missing field."""
        condition = ValueCondition(field_name="query_type", allowed={"coding"})
        assert condition.matches({}) is False

    def test_none_value(self):
        """Test behavior with None value."""
        condition = ValueCondition(field_name="query_type", allowed={"coding"})
        assert condition.matches({"query_type": None}) is False

    def test_numeric_value(self):
        """Test with numeric values converted to strings."""
        condition = ValueCondition(field_name="status_code", allowed={"200", "201"})
        assert condition.matches({"status_code": 200}) is True


class TestMinMaxCondition:
    """Test suite for MinMaxCondition."""

    def test_value_within_range(self):
        """Test matching when value is within range."""
        condition = MinMaxCondition(field_name="age", min_value=18, max_value=65)
        assert condition.matches({"age": 25}) is True
        assert condition.matches({"age": 18}) is True  # Boundary
        assert condition.matches({"age": 65}) is True  # Boundary

    def test_value_below_min(self):
        """Test not matching when value is below minimum."""
        condition = MinMaxCondition(field_name="age", min_value=18)
        assert condition.matches({"age": 17}) is False
        assert condition.matches({"age": 0}) is False

    def test_value_above_max(self):
        """Test not matching when value is above maximum."""
        condition = MinMaxCondition(field_name="age", max_value=65)
        assert condition.matches({"age": 66}) is False
        assert condition.matches({"age": 100}) is False

    def test_only_min(self):
        """Test with only minimum constraint."""
        condition = MinMaxCondition(field_name="age", min_value=18)
        assert condition.matches({"age": 18}) is True
        assert condition.matches({"age": 100}) is True
        assert condition.matches({"age": 17}) is False

    def test_only_max(self):
        """Test with only maximum constraint."""
        condition = MinMaxCondition(field_name="age", max_value=65)
        assert condition.matches({"age": 0}) is True
        assert condition.matches({"age": 65}) is True
        assert condition.matches({"age": 66}) is False

    def test_missing_field_not_allowed(self):
        """Test missing field not allowed (default)."""
        condition = MinMaxCondition(field_name="age", min_value=18)
        assert condition.matches({}) is False

    def test_missing_field_allowed(self):
        """Test missing field allowed."""
        condition = MinMaxCondition(field_name="age", min_value=18, allow_missing=True)
        assert condition.matches({}) is True

    def test_non_numeric_value(self):
        """Test behavior with non-numeric value."""
        condition = MinMaxCondition(field_name="age", min_value=18)
        assert condition.matches({"age": "invalid"}) is False

    def test_float_values(self):
        """Test with float values."""
        condition = MinMaxCondition(field_name="score", min_value=0.5, max_value=1.0)
        assert condition.matches({"score": 0.75}) is True
        assert condition.matches({"score": 0.5}) is True
        assert condition.matches({"score": 1.0}) is True
        assert condition.matches({"score": 0.49}) is False

    def test_string_numeric_conversion(self):
        """Test conversion of string numeric values."""
        condition = MinMaxCondition(field_name="age", min_value=18, max_value=65)
        assert condition.matches({"age": "25"}) is True
        assert condition.matches({"age": "17"}) is False


class TestHashCondition:
    """Test suite for HashCondition."""

    def test_consistent_hashing(self):
        """Test that same value always produces same result."""
        condition = HashCondition(hash_key="session_id", percentage=50)

        # Same session_id should always produce same result
        result1 = condition.matches({"session_id": "user123"})
        result2 = condition.matches({"session_id": "user123"})
        assert result1 == result2

    def test_percentage_zero(self):
        """Test with 0% percentage."""
        condition = HashCondition(hash_key="session_id", percentage=0)
        assert condition.matches({"session_id": "user123"}) is False

    def test_percentage_hundred(self):
        """Test with 100% percentage."""
        condition = HashCondition(hash_key="session_id", percentage=100)
        assert condition.matches({"session_id": "user123"}) is True

    def test_percentage_distribution(self):
        """Test that percentage roughly matches distribution."""
        condition = HashCondition(hash_key="session_id", percentage=30)

        # Test with 1000 different session IDs
        matches = 0
        total = 1000
        for i in range(total):
            if condition.matches({"session_id": f"session_{i}"}):
                matches += 1

        # Should be roughly 30%, allow 5% margin
        ratio = matches / total * 100
        assert 25 <= ratio <= 35

    def test_fine_grained_percentage(self):
        """Test fine-grained percentage (decimal places)."""
        condition = HashCondition(hash_key="session_id", percentage=99.99)

        # Should match almost all sessions
        matches = 0
        total = 1000
        for i in range(total):
            if condition.matches({"session_id": f"session_{i}"}):
                matches += 1

        # Should be very high percentage
        ratio = matches / total * 100
        assert ratio >= 99.5

    def test_missing_hash_key(self):
        """Test behavior with missing hash key."""
        condition = HashCondition(hash_key="session_id", percentage=50)
        assert condition.matches({}) is False

    def test_none_hash_value(self):
        """Test behavior with None hash value."""
        condition = HashCondition(hash_key="session_id", percentage=50)
        assert condition.matches({"session_id": None}) is False

    def test_invalid_percentage_negative(self):
        """Test with negative percentage."""
        condition = HashCondition(hash_key="session_id", percentage=-10)
        assert condition.matches({"session_id": "user123"}) is False

    def test_invalid_percentage_over_hundred(self):
        """Test with percentage over 100."""
        condition = HashCondition(hash_key="session_id", percentage=150)
        assert condition.matches({"session_id": "user123"}) is False

    def test_different_hash_keys(self):
        """Test that different hash keys produce different distributions."""
        condition_session = HashCondition(hash_key="session_id", percentage=50)
        condition_user = HashCondition(hash_key="user_id", percentage=50)

        ctx = {"session_id": "abc123", "user_id": "abc123"}
        # Even with same value, different keys may produce different results
        # (though not guaranteed, depends on hash function)
        result_session = condition_session.matches(ctx)
        result_user = condition_user.matches(ctx)

        # Just verify both produce boolean results
        assert isinstance(result_session, bool)
        assert isinstance(result_user, bool)

    def test_numeric_hash_values(self):
        """Test with numeric hash values."""
        condition = HashCondition(hash_key="user_id", percentage=50)

        # Numeric values should be converted to string for hashing
        result1 = condition.matches({"user_id": 12345})
        result2 = condition.matches({"user_id": 12345})
        assert result1 == result2  # Consistency check


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
