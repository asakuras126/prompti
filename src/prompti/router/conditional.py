"""Conditional routing pipeline with sequential conditions and candidate selection."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from ..logger import get_logger

logger = get_logger(__name__)

RouteContext = Mapping[str, Any]


@runtime_checkable
class Condition(Protocol):
    """Interface that all routing conditions must implement."""

    def matches(self, ctx: RouteContext) -> bool:
        """Return True when the condition is satisfied by the context."""
        ...


@dataclass(slots=True)
class ListCondition:
    """Whitelist/blacklist matcher for an arbitrary context field."""

    field_name: str
    allow: set[str] = field(default_factory=set)
    deny: set[str] = field(default_factory=set)

    def matches(self, ctx: RouteContext) -> bool:
        """Return True when the given context satisfies the allow/deny lists."""
        value = (ctx.get(self.field_name) or '').lower()
        if self.allow and value not in self.allow:
            return False
        if self.deny and value in self.deny:
            return False
        return True


@dataclass(slots=True)
class BoolCondition:
    """Boolean flag matcher."""

    field_name: str
    expected: bool

    def matches(self, ctx: RouteContext) -> bool:
        """Return True if the boolean field matches the expected value."""
        return bool(ctx.get(self.field_name)) is self.expected


@dataclass(slots=True)
class ValueCondition:
    """String/enum matcher requiring the field to equal one of the allowed values."""

    field_name: str
    allowed: set[str] = field(default_factory=set)
    case_sensitive: bool = False

    def matches(self, ctx: RouteContext) -> bool:
        """Return True if the context value exists in the allowed set."""
        raw_value = ctx.get(self.field_name)
        if raw_value is None:
            return False
        value = str(raw_value)
        if not self.case_sensitive:
            value = value.lower()
        return value in self.allowed


@dataclass(slots=True)
class MinMaxCondition:
    """Numeric field matcher enforcing minimum/maximum bounds."""

    field_name: str
    min_value: float | None = None
    max_value: float | None = None
    allow_missing: bool = False

    def matches(self, ctx: RouteContext) -> bool:
        """Return True if the numeric value stays within the configured limits."""
        raw_value = ctx.get(self.field_name)
        if raw_value is None:
            return self.allow_missing

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return False

        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True


@dataclass(slots=True)
class HashCondition:
    """Hash-based percentage matcher for traffic splitting.

    Uses consistent hashing to match a percentage of sessions.
    Useful for A/B testing and gradual rollouts.

    NOTE: When you adjust the percentage, some sessions will be re-routed.
    For example:
        - percentage=30: session_123 (bucket=25) → matched
        - percentage=20: session_123 (bucket=25) → NOT matched  # Changed!

    This is expected behavior when adjusting traffic percentages.

    The hash bucket uses 10000 buckets (0.01% precision) to support fine-grained
    percentage control like 99.9%, 50.5%, etc.

    Example:
        HashCondition(hash_key='session_id', percentage=30.5)
        # Matches ~30.5% of unique session_ids consistently

        HashCondition(hash_key='session_id', percentage=99.99)
        # Matches ~99.99% of unique session_ids consistently
    """

    hash_key: str
    percentage: float  # 0-100, supports up to 2 decimal places (0.01% precision)

    def matches(self, ctx: RouteContext) -> bool:
        """Return True if hash(key) falls within the percentage threshold."""
        if not (0 <= self.percentage <= 100):
            return False

        hash_value = ctx.get(self.hash_key)
        if hash_value is None:
            return False

        # Use MD5 for consistent hashing
        hash_str = str(hash_value)
        hash_bytes = hashlib.md5(hash_str.encode('utf-8')).digest()
        hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')

        # Map to 1-10000 range for 0.01% precision
        bucket = (hash_int % 10000) + 1  # 1-10000

        # Convert percentage to bucket threshold (percentage * 100)
        # e.g., 99.9% → 9990 buckets, 50.5% → 5050 buckets
        threshold = self.percentage * 100

        return bucket <= threshold



@dataclass(slots=True)
class ConditionConfig:
    """Aggregates parsed conditions before converting to a flat list."""

    conditions: list[Condition] = field(default_factory=list)

    def add(self, condition: Condition) -> None:
        """Append a condition, ensuring it implements the protocol."""
        if not isinstance(condition, Condition):
            raise TypeError('Invalid condition object provided.')
        self.conditions.append(condition)

    def extend(self, extra_conditions: Iterable[Condition]) -> None:
        """Append multiple conditions with validation."""
        for condition in extra_conditions:
            self.add(condition)

    def as_conditions(self) -> list[Condition]:
        """Return conditions in insertion order."""
        return list(self.conditions)


@dataclass(slots=True)
class Candidate:
    """Destination that the router can select."""

    name: str
    weight: int = 1
    load: int = 0


class Selector:
    """Base class for candidate selection strategies."""

    def select(self, ctx: RouteContext) -> Candidate:
        """Select a candidate for the provided context."""
        raise NotImplementedError

    def record(self, candidate: Candidate, tokens: int | None = None) -> None:
        """Hook to signal that a candidate was used."""
        return None


class WRRSelector(Selector):
    """Deterministic selector using weighted round robin."""

    def __init__(self, candidates: Sequence[Candidate]) -> None:
        """Precompute the weighted schedule for all candidates."""
        if not candidates:
            raise ValueError('WeightedRoundRobinSelector requires at least one candidate.')

        self._schedule: list[Candidate] = []
        for candidate in candidates:
            self._schedule.extend([candidate] * max(candidate.weight, 1))

        self._lock = threading.Lock()
        self._cursor = 0

    def select(self, ctx: RouteContext) -> Candidate:  # noqa: ARG002
        """Return the next candidate in the round-robin schedule."""
        with self._lock:
            candidate = self._schedule[self._cursor]
            self._cursor = (self._cursor + 1) % len(self._schedule)
            return candidate


@dataclass
class Route:
    """Single routing pipeline entry."""

    name: str
    conditions: list[Condition]
    selector: Selector
    weight: int = 0
    allow_exceed_weight: bool = False
    enable_sticky: bool = False

    def matches(self, ctx: RouteContext) -> bool:
        """Return True when every condition matches the context."""
        for condition in self.conditions:
            if not condition.matches(ctx):
                return False
        return True

    def get_candidate_names(self) -> set[str]:
        """Return all candidate names for this route."""
        if isinstance(self.selector, WRRSelector):
            return {candidate.name for candidate in self.selector._schedule}
        return set()


class RoutePipeline:
    """Evaluates routes in order and selects the first matching candidate."""

    def __init__(
        self,
        routes: Sequence[Route],
        sticky_config: dict[str, Any] | None = None,
        template_name: str | None = None,
        global_config_getter: Any = None,
        redis_client: Any = None
    ) -> None:
        """Store the ordered routes and initialize quota bookkeeping."""
        if not routes:
            raise ValueError('RoutePipeline requires at least one route.')
        self._routes = list(routes)
        self._quota_template = {
            route.name: route.weight for route in self._routes if route.weight > 0
        }
        self._quota_remaining = dict(self._quota_template)
        self._quota_lock = threading.Lock()

        # Sticky routing support (Redis-based, for cross-pod sharing)
        self._sticky_config = sticky_config
        self._template_name = template_name
        self._global_config_getter = global_config_getter
        self._redis_client = redis_client

        # In-memory routing history for first-time detection (metrics only)
        # Key: (session_id, template_name, route_name) → variant_name
        # Max size: 10000 entries, LRU eviction
        from collections import OrderedDict
        self._routing_history: OrderedDict[tuple[str, str, str], str] = OrderedDict()
        self._routing_history_lock = threading.Lock()
        self._routing_history_max_size = 10000

    def _consume_quota(self, route: Route) -> None:
        if route.weight <= 0:
            return
        with self._quota_lock:
            if not self._quota_remaining:
                return
            remaining = self._quota_remaining.get(route.name, 0)
            if remaining > 0:
                self._quota_remaining[route.name] = remaining - 1

            if all(value <= 0 for value in self._quota_remaining.values()):
                self._quota_remaining = dict(self._quota_template)

    def select(self, ctx: RouteContext) -> Candidate:
        """Return the first candidate whose route matches the context."""
        for route in self._routes:
            # Step 1: Check all conditions (business logic first)
            if not route.matches(ctx):
                continue
            if (
                route.weight > 0
                and not route.allow_exceed_weight
                and self._quota_remaining.get(route.name, route.weight) <= 0
            ):
                continue

            # Step 2: Check if this is first-time routing (in-memory, for metrics)
            is_first_routing = self._is_first_routing(ctx, route)

            # Step 3: Check sticky routing if enabled (Redis-based, for actual routing)
            candidate = None
            routing_source = 'normal'

            if route.enable_sticky and self._sticky_config:
                sticky_variant_name = self._get_sticky_variant(ctx, route)
                if sticky_variant_name:
                    # Validate sticky variant is still a valid candidate for this route
                    if sticky_variant_name in route.get_candidate_names():
                        candidate = Candidate(name=sticky_variant_name)
                        routing_source = 'sticky'
                        logger.info(
                            f"[Variant Router] Sticky hit for route '{route.name}': "
                            f"variant '{sticky_variant_name}'"
                        )
                    else:
                        logger.debug(
                            f"[Variant Router] Sticky variant '{sticky_variant_name}' "
                            f"no longer valid for route '{route.name}', re-routing"
                        )
                        # Clear invalid sticky variant
                        self._clear_sticky_variant(ctx, route)

            # Step 4: No valid sticky variant, select normally
            if candidate is None:
                candidate = route.selector.select(ctx)
                route.selector.record(candidate, ctx.get('token_usage'))
                self._consume_quota(route)

                # Store sticky variant if enabled (Redis, for cross-pod sharing)
                if route.enable_sticky and self._sticky_config:
                    self._store_sticky_variant(ctx, route, candidate.name)

            # Step 5: Record routing history in memory (for is_first_routing tracking)
            self._record_routing_history(ctx, route, candidate.name)

            # Log the matched route and selected candidate
            logger.info(
                f"[Variant Router] Matched rule: '{route.name}' → selected variant: '{candidate.name}' "
                f"(source: {routing_source})"
            )
            logger.debug(f"[Variant Router] Full context: {ctx}")

            # Record variant routing metrics for conditional router pipeline
            try:
                from ..otel import get_llm_metrics
                llm_metrics = get_llm_metrics()
                if llm_metrics and llm_metrics.enabled:
                    template_name = ctx.get('template_name', '')
                    llm_metrics.record_variant_routing(
                        template_name=template_name,
                        variant_name=candidate.name,
                        route_name=route.name,
                        routing_source=routing_source,
                        is_first_routing=str(is_first_routing).lower()  # "true" or "false"
                    )
            except Exception as e:
                logger.debug(f"Failed to record variant routing metrics: {e}")

            return candidate

        raise LookupError('No route matched the provided context.')

    def _get_sticky_variant(self, ctx: RouteContext, route: Route) -> str | None:
        """Get sticky variant from Redis for this route."""
        if not self._sticky_config:
            return None

        try:
            from .sticky_helper import StickyRoutingHelper
            return StickyRoutingHelper.get_sticky_variant(
                sticky_config=self._sticky_config,
                route_context=ctx,
                template_name=self._template_name or '',
                route_name=route.name,
                global_config_getter=self._global_config_getter,
                redis_client=self._redis_client
            )
        except Exception as e:
            logger.warning(f"Failed to get sticky variant: {e}")
            return None

    def _store_sticky_variant(self, ctx: RouteContext, route: Route, variant_name: str) -> None:
        """Store sticky variant to Redis for this route."""
        if not self._sticky_config:
            return

        try:
            from .sticky_helper import StickyRoutingHelper
            StickyRoutingHelper.store_sticky_variant(
                sticky_config=self._sticky_config,
                route_context=ctx,
                template_name=self._template_name or '',
                route_name=route.name,
                variant_name=variant_name,
                global_config_getter=self._global_config_getter,
                redis_client=self._redis_client
            )
        except Exception as e:
            logger.warning(f"Failed to store sticky variant: {e}")

    def _clear_sticky_variant(self, ctx: RouteContext, route: Route) -> None:
        """Clear sticky variant from Redis for this route."""
        if not self._sticky_config:
            return

        try:
            from .sticky_helper import StickyRoutingHelper
            session_id = ctx.get('session_id')
            if session_id:
                StickyRoutingHelper.clear_sticky_variant(
                    sticky_config=self._sticky_config,
                    session_id=session_id,
                    template_name=self._template_name or '',
                    route_name=route.name,
                    global_config_getter=self._global_config_getter
                )
        except Exception as e:
            logger.warning(f"Failed to clear sticky variant: {e}")

    def _is_first_routing(self, ctx: RouteContext, route: Route) -> bool:
        """Check if this is the first routing for this session+template+route (in-memory).

        Uses LRU cache in memory for fast lookup, not persisted across restarts.
        Only used for metrics tracking, does not affect routing logic.

        Returns:
            True if first routing for this combination, False otherwise
        """
        session_id = ctx.get('session_id')
        if not session_id:
            return True  # No session_id, consider as first-time

        template_name = self._template_name or ctx.get('template_name', '')
        cache_key = (session_id, template_name, route.name)

        with self._routing_history_lock:
            return cache_key not in self._routing_history

    def _record_routing_history(self, ctx: RouteContext, route: Route, variant_name: str) -> None:
        """Record routing history in memory for first-time detection (metrics only).

        Uses LRU cache with max size limit. Does not affect routing logic.

        Args:
            ctx: Route context
            route: Matched route
            variant_name: Selected variant name
        """
        session_id = ctx.get('session_id')
        if not session_id:
            return

        template_name = self._template_name or ctx.get('template_name', '')
        cache_key = (session_id, template_name, route.name)

        with self._routing_history_lock:
            # Update or add entry (move to end for LRU)
            if cache_key in self._routing_history:
                self._routing_history.move_to_end(cache_key)
            else:
                self._routing_history[cache_key] = variant_name

            # Evict oldest entry if cache is full
            if len(self._routing_history) > self._routing_history_max_size:
                self._routing_history.popitem(last=False)  # Remove oldest (FIFO)


def build_route_conditions(data: dict[str, Any]) -> list[Condition]:
    """Construct an ordered list of Conditions from a plain dictionary."""

    def _normalize(values: Iterable[str] | str | None) -> set[str]:
        if values is None:
            return set()
        if isinstance(values, str):
            return {values.lower()}
        return {str(value).lower() for value in values}

    def _coerce_numeric(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _iter_values(values: Iterable[str] | str | None) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str):
            return [values]
        return list(values)

    config = ConditionConfig()
    condition_specs: list[tuple[str, dict[str, Any]]] = []

    for spec in data.get('condition_list', []):
        spec_type = spec.get('type')
        if spec_type not in {'list', 'bool', 'value', 'min_max', 'hash'}:
            continue
        condition_specs.append((spec_type, spec))

    query_type = data.get('query_type')
    if query_type is not None:
        condition_specs.append(('value', {'field': 'query_type', 'values': query_type}))

    for cond_type, spec in condition_specs:
        spec = dict(spec)  # avoid mutating caller-provided dicts
        field_name = spec.get('field')

        # Hash condition doesn't require 'field', skip the check for it
        if not field_name and cond_type != 'hash':
            continue

        if cond_type == 'list':
            config.add(
                ListCondition(
                    field_name=field_name,
                    allow=_normalize(spec.get('allow')),
                    deny=_normalize(spec.get('deny')),
                )
            )
        elif cond_type == 'bool':
            expected = spec.get('expected')
            if expected is None:
                continue
            config.add(BoolCondition(field_name=field_name, expected=bool(expected)))
        elif cond_type == 'value':
            case_sensitive = bool(spec.get('case_sensitive', False))
            iterable = _iter_values(spec.get('values'))
            normalized = (
                {str(v) for v in iterable}
                if case_sensitive
                else {str(v).lower() for v in iterable}
            )
            if not normalized:
                continue
            config.add(
                ValueCondition(
                    field_name=field_name,
                    allowed=normalized,
                    case_sensitive=case_sensitive,
                )
            )
        elif cond_type == 'min_max':
            min_value = _coerce_numeric(spec.get('min'))
            max_value = _coerce_numeric(spec.get('max'))
            if min_value is None and max_value is None:
                continue
            allow_missing = bool(spec.get('allow_missing', False))
            config.add(
                MinMaxCondition(
                    field_name=field_name,
                    min_value=min_value,
                    max_value=max_value,
                    allow_missing=allow_missing,
                )
            )
        elif cond_type == 'hash':
            hash_key = spec.get('hash_key', 'session_id')
            percentage = _coerce_numeric(spec.get('percentage'))
            if percentage is None:
                continue
            config.add(
                HashCondition(
                    hash_key=hash_key,
                    percentage=percentage,
                )
            )

    return config.as_conditions()


def build_selector(
    selector_cfg: dict[str, Any],
    candidates: Sequence[Candidate],
) -> Selector:
    """Instantiate a selector from config values."""
    selector_type = (selector_cfg or {}).get('type', 'weighted_round_robin')
    if selector_type != 'weighted_round_robin':
        raise ValueError(f"Unsupported selector type '{selector_type}'.")
    return WRRSelector(candidates)


def build_pipeline(
    route_definitions: Iterable[dict[str, Any]],
    sticky_config: dict[str, Any] | None = None,
    template_name: str | None = None,
    global_config_getter: Any = None,
    redis_client: Any = None
) -> RoutePipeline:
    """Create a pipeline from declarative dictionaries.

    Args:
        route_definitions: Route configuration dictionaries
        sticky_config: Sticky routing configuration (optional)
        template_name: Template name for sticky routing (optional)
        global_config_getter: Global config getter function (optional)
        redis_client: Redis client instance (optional)

    Returns:
        Configured RoutePipeline instance
    """
    routes: list[Route] = []
    for route_cfg in route_definitions:
        candidates = [
            Candidate(name=item['name'], weight=item.get('weight', 1))
            for item in route_cfg.get('candidates', [])
        ]
        if not candidates:
            raise ValueError(f"Route '{route_cfg.get('name')}' has no candidates.")

        selector = build_selector(route_cfg.get('selector', {}), candidates)
        raw_conditions = route_cfg.get('conditions', {})
        if isinstance(raw_conditions, list):
            condition_payload = {'condition_list': raw_conditions}
        elif raw_conditions is None:
            condition_payload = {}
        else:
            condition_payload = dict(raw_conditions)
        conditions = build_route_conditions(condition_payload)

        route = Route(
            name=route_cfg.get('name', 'route'),
            conditions=conditions,
            selector=selector,
            weight=route_cfg.get('weight', 0),
            allow_exceed_weight=route_cfg.get('allow_exceed_weight', False),
            enable_sticky=route_cfg.get('enable_sticky', False),
        )
        routes.append(route)

    return RoutePipeline(
        routes=routes,
        sticky_config=sticky_config,
        template_name=template_name,
        global_config_getter=global_config_getter,
        redis_client=redis_client
    )
