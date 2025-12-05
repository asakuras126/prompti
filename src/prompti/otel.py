#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
OpenTelemetry integration for LLM metrics tracking.

This module provides instrumentation for tracking LLM-related metrics including:
- Call counts and latency
- Token consumption (input/output/total)
- Error rates and types
- Model performance metrics

Features async queue for non-blocking metrics reporting.
"""

import time
import logging
import socket
import asyncio
import threading
import os
from queue import Queue, Empty
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from opentelemetry import trace, metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics._internal.instrument import Counter, Histogram
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, AggregationTemporality
from opentelemetry.sdk.metrics.view import View
from opentelemetry.sdk.metrics._internal.aggregation import ExplicitBucketHistogramAggregation
from opentelemetry.sdk.resources import SERVICE_NAME, HOST_NAME, Resource

# Setup logger first
from .logger import get_logger

logger = get_logger(__name__)

# Try to import OTLP exporters, fall back to console exporter
try:
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    OTLP_AVAILABLE = True
except ImportError:
    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        OTLP_AVAILABLE = True
    except ImportError:
        try:
            from opentelemetry.sdk.metrics.export import ConsoleMetricExporter as OTLPMetricExporter
            OTLP_AVAILABLE = False
            logger.warning("OTLP exporters not available, using ConsoleMetricExporter for demonstration")
        except ImportError:
            # Create a no-op exporter
            class NoOpMetricExporter:
                def __init__(self, *args, **kwargs):
                    pass
                def export(self, *args, **kwargs):
                    return None
                def shutdown(self, *args, **kwargs):
                    return None
            
            OTLPMetricExporter = NoOpMetricExporter
            OTLP_AVAILABLE = False
            logger.warning("No metric exporters available, using no-op exporter")


@dataclass
class MetricEvent:
    """Base class for metric events."""
    timestamp: float
    attributes: Dict[str, str]


@dataclass
class RequestStartEvent(MetricEvent):
    """Event for LLM request start."""
    pass


@dataclass
class RequestCompleteEvent(MetricEvent):
    """Event for LLM request completion."""
    duration_ms: float  # 总耗时（毫秒）
    first_token_latency_ms: float  # 首包延迟（毫秒）
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_tokens: int = 0  # 缓存读取tokens
    cache_write_tokens: int = 0  # 缓存写入tokens
    success: bool = True
    error_type: str = ""


@dataclass
class StreamingChunkEvent(MetricEvent):
    """Event for streaming chunk."""
    chunk_size: int
    chunk_tokens: int


@dataclass
class VariantRoutingEvent(MetricEvent):
    """Event for variant routing decision."""
    template_name: str
    variant_name: str
    route_name: str
    routing_source: str
    extra_attributes: dict[str, str] | None = None


class AsyncMetricsQueue:
    """Async queue for buffering metrics events without blocking LLM requests."""
    
    def __init__(self, max_size: int = 10000):
        self.queue = Queue(maxsize=max_size)
        self.running = False
        self.worker_thread = None
        self.loop = None
        
    def start(self, llm_metrics: 'LLMMetrics'):
        """Start the background worker thread."""
        if self.running:
            return
            
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._worker,
            args=(llm_metrics,),
            daemon=True,
            name="otel-metrics-worker"
        )
        self.worker_thread.start()
        logger.debug("OTEL metrics async queue started")
    
    def stop(self):
        """Stop the background worker thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info("OTEL metrics async queue stopped")
    
    def put_nowait(self, event: MetricEvent):
        """Add event to queue without blocking."""
        try:
            self.queue.put_nowait(event)
        except Exception as e:
            # Drop the event if queue is full to avoid blocking
            logger.warning(f"Failed to queue metric event (queue full?): {e}")
    
    def _worker(self, llm_metrics: 'LLMMetrics'):
        """Background worker that processes queued events."""
        while self.running:
            try:
                # Get event with timeout to allow checking running status
                event = self.queue.get(timeout=1.0)
                
                # Process the event
                try:
                    self._process_event(event, llm_metrics)
                except Exception as e:
                    logger.error(f"Error processing metric event: {e}")
                finally:
                    self.queue.task_done()
                    
            except Empty:
                # Timeout - continue loop to check running status
                continue
            except Exception as e:
                logger.error(f"Error in metrics worker: {e}")
        
        # Process remaining events before shutdown
        self._drain_queue(llm_metrics)
    
    def _drain_queue(self, llm_metrics: 'LLMMetrics'):
        """Process all remaining events in queue during shutdown."""
        while True:
            try:
                event = self.queue.get_nowait()
                self._process_event(event, llm_metrics)
                self.queue.task_done()
            except Empty:
                break
            except Exception as e:
                logger.error(f"Error draining metric event: {e}")
    
    def _process_event(self, event: MetricEvent, llm_metrics: 'LLMMetrics'):
        """Process a single metric event."""
        try:
            if isinstance(event, RequestStartEvent):
                # 在worker线程中打印处理日志
                logger.debug(f"OTEL处理 - 请求开始指标: model={event.attributes.get('model', 'unknown')}")
                llm_metrics._record_request_start_sync(event.attributes)
            elif isinstance(event, RequestCompleteEvent):
                # 在worker线程中打印详细的处理日志
                status = "成功" if event.success else f"失败({event.error_type})"

                # 构建 tokens 字符串
                tokens_str = (f"tokens(prompt={event.input_tokens}, completion={event.output_tokens},"
                              f" total={event.total_tokens}")
                if event.cache_read_tokens > 0 or event.cache_write_tokens > 0:
                    tokens_str += f", cache_read={event.cache_read_tokens}, cache_write={event.cache_write_tokens}"
                tokens_str += ")"

                log_parts = [
                    f"OTEL - model={event.attributes.get('model', 'unknown')}",
                    f"template={event.attributes.get('template_name', 'unknown')}",
                    f"status={status}",
                    f"duration={event.duration_ms:.1f}ms",
                    f"ttft={event.first_token_latency_ms:.1f}ms" if event.first_token_latency_ms > 0
                    else "ttft=N/A",
                    tokens_str
                ]

                logger.info(" | ".join(log_parts))
                llm_metrics._record_request_complete_sync(
                    event.attributes,
                    event.duration_ms,
                    event.first_token_latency_ms,
                    event.input_tokens,
                    event.output_tokens,
                    event.total_tokens,
                    event.cache_read_tokens,
                    event.cache_write_tokens,
                    event.success,
                    event.error_type
                )
            elif isinstance(event, VariantRoutingEvent):
                logger.debug(f"OTEL处理 - Variant路由指标: variant={event.variant_name}, route={event.route_name}")
                llm_metrics._record_variant_routing_sync(
                    event.template_name,
                    event.variant_name,
                    event.route_name,
                    event.routing_source,
                    **(event.extra_attributes or {})
                )
        except Exception as e:
            logger.error(f"Error processing {type(event).__name__}: {e}")


class LLMMetrics:
    """LLM metrics tracker using OpenTelemetry."""

    def __init__(
        self,
        service_name: str = "prompti",
        endpoint: str = "http://localhost:4317",
        export_interval_ms: int = 5000,
        enabled: bool = True,
        queue_size: int = 10000
    ):
        """Initialize LLM metrics tracker.
        
        Args:
            service_name: Service name for metrics
            endpoint: OTEL collector endpoint
            export_interval_ms: Metrics export interval in milliseconds
            enabled: Whether metrics tracking is enabled
            queue_size: Max size of async metrics queue
        """
        self.enabled = enabled
        self.service_name = service_name
        self.endpoint = endpoint

        # Get pod_name from environment variable for multi-pod scenarios
        # This ensures each pod has a unique identifier in metrics labels
        appspace_env = os.getenv('APPSPACE_ENV', 'local')
        appspace_pod_name = os.getenv('APPSPACE_POD_NAME', 'local')
        self.pod_name = f"{appspace_env}-{appspace_pod_name}"

        if not self.enabled:
            logger.info("LLM metrics tracking is disabled")
            self.metrics_queue = None
            return

        # Create resource
        self.resource = Resource.create({
            SERVICE_NAME: service_name,
            HOST_NAME: socket.gethostname()
        })
        
        # Setup metrics provider
        self._setup_metrics_provider(export_interval_ms)
        
        # Initialize metrics instruments
        self._init_metrics()
        
        # Initialize async queue
        self.metrics_queue = AsyncMetricsQueue(max_size=queue_size)
        self.metrics_queue.start(self)

        logger.debug(f"LLM metrics initialized - service: {service_name},"
                    f" endpoint: {endpoint}, queue_size: {queue_size}, pod_name: {self.pod_name}")
    
    def _setup_metrics_provider(self, export_interval_ms: int):
        """Setup OpenTelemetry metrics provider."""
        try:
            if OTLP_AVAILABLE:
                # Use OTLP exporter with endpoint
                # IMPORTANT: Set preferred_temporality to CUMULATIVE for Prometheus compatibility
                # Prometheus expects cumulative counters and histograms, not delta
                exporter = OTLPMetricExporter(
                    endpoint=self.endpoint,
                    insecure=True,
                    preferred_temporality={
                        Counter: AggregationTemporality.CUMULATIVE,
                        Histogram: AggregationTemporality.CUMULATIVE
                    }
                )
            else:
                # Use console or no-op exporter
                exporter = OTLPMetricExporter()

            metric_reader = PeriodicExportingMetricReader(
                exporter,
                export_interval_millis=export_interval_ms
            )

            # Define custom bucket boundaries for different histogram metrics
            # Bucket boundaries in milliseconds: 500ms, 1s, 2s, 3s, 5s, 7s, 10s, 15s, 20s, 30s, 45s, 60s, 90s,
            #  120s, 180s, 300s, 600s, 1800s, 3600s
            # 这些bucket更适合LLM请求的延迟分布（大多数在1-60秒之间）
            # 添加更大的 bucket 以覆盖 coding_agent 等长时间任务（最大到 1 小时）
            duration_buckets = [500, 1000, 2000, 3000, 5000, 7000, 10000,
                                15000, 20000, 30000, 45000, 60000, 90000, 120000, 180000,
                                300000, 600000, 1800000, 3600000]

            # TTFT通常更快，使用更细粒度的bucket（毫秒）
            ttft_buckets = [100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000]

            # Token count buckets: 优化为LLM常见的token范围
            token_buckets = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]

            # Create views to configure custom bucket boundaries for histograms
            views = [
                View(
                    instrument_name="llm_request_duration_milliseconds",
                    aggregation=ExplicitBucketHistogramAggregation(boundaries=duration_buckets)
                ),
                View(
                    instrument_name="llm_first_token_latency_milliseconds",
                    aggregation=ExplicitBucketHistogramAggregation(boundaries=ttft_buckets)
                ),
                View(
                    instrument_name="llm_tokens_per_request",
                    aggregation=ExplicitBucketHistogramAggregation(boundaries=token_buckets)
                ),
            ]

            meter_provider = MeterProvider(
                resource=self.resource,
                metric_readers=[metric_reader],
                views=views
            )

            metrics.set_meter_provider(meter_provider)
            self.meter = meter_provider.get_meter("prompti.llm", version="1.0.0")

            logger.debug(f"OpenTelemetry metrics provider initialized with "
                        f"CUMULATIVE temporality for Prometheus compatibility")

        except Exception as e:
            logger.error(f"Failed to setup metrics provider: {e}")
            self.enabled = False
            raise
    
    def _init_metrics(self):
        """Initialize all metric instruments."""
        if not self.enabled:
            return
            
        try:
            # Counters
            self.llm_requests_total = self.meter.create_counter(
                name="llm_requests_total",
                description="Total number of LLM requests",
                unit="1"
            )
            
            self.llm_tokens_total = self.meter.create_counter(
                name="llm_tokens_total",
                description="Total number of tokens consumed by type (prompt_tokens, completion_tokens,"
                            " cache_read_input_tokens, cache_creation_input_tokens, total_tokens)",
                unit="1"
            )
            
            self.llm_errors_total = self.meter.create_counter(
                name="llm_errors_total",
                description="Total number of LLM errors",
                unit="1"
            )
            
            # Histograms - custom buckets are configured via Views in _setup_metrics_provider
            self.llm_request_duration = self.meter.create_histogram(
                name="llm_request_duration_milliseconds",
                description="LLM request total duration in milliseconds",
                unit="ms"
            )

            self.llm_first_token_latency = self.meter.create_histogram(
                name="llm_first_token_latency_milliseconds",
                description="Time to first token (TTFT) in milliseconds",
                unit="ms"
            )

            self.llm_tokens_per_request = self.meter.create_histogram(
                name="llm_tokens_per_request",
                description="Distribution of total tokens per request (for P50/P90/P95/P99 analysis)",
                unit="1"
            )
            
            # Gauges for current state
            self.llm_active_requests = self.meter.create_up_down_counter(
                name="llm_active_requests",
                description="Number of active LLM requests",
                unit="1"
            )

            # Variant selection metrics
            self.variant_selection_duration = self.meter.create_histogram(
                name="variant_selection_duration_milliseconds",
                description="Time spent selecting variant in milliseconds",
                unit="ms"
            )

            # Variant routing metrics
            self.variant_routing_total = self.meter.create_counter(
                name="variant_routing_total",
                description="Total number of variant routing decisions by template, variant,"
                            " and route_name from conditional router pipeline",
                unit="1"
            )

            logger.debug("LLM metrics instruments initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize metrics instruments: {e}")
            self.enabled = False
            raise
    
    def record_request_start(
        self,
        template_name: str = "",
        model: str = "",
        provider: str = "",
        **extra_attributes
    ) -> Dict[str, Any]:
        """Record the start of an LLM request (async, non-blocking).
        
        Args:
            template_name: Name of the prompt template
            model: Model name
            provider: Model provider (e.g., openai, claude)
            **extra_attributes: Additional attributes for metrics
            
        Returns:
            Dict containing request context for completion tracking
        """
        if not self.enabled or not self.metrics_queue:
            return {}
            
        start_time = time.time()

        # Base attributes
        attributes = {
            "template_name": template_name,
            "model": model,
            "provider": provider,
            "pod_name": self.pod_name,  # Add pod_name to distinguish metrics from different pods
        }

        # Add extra attributes
        attributes.update(extra_attributes)

        # Filter out empty values
        attributes = {k: v for k, v in attributes.items() if v}
        
        # Queue event asynchronously (non-blocking)
        event = RequestStartEvent(
            timestamp=start_time,
            attributes=attributes
        )
        self.metrics_queue.put_nowait(event)

        logger.debug("Queued LLM request start")
        
        return {
            "start_time": start_time,
            "attributes": attributes
        }
    
    def _record_request_start_sync(self, attributes: Dict[str, str]):
        """Synchronously record request start metrics (called by worker thread)."""
        try:
            # Increment active requests
            self.llm_active_requests.add(1, attributes)

            # Increment total requests
            self.llm_requests_total.add(1, attributes)

        except Exception as e:
            logger.error(f"Failed to record request start metrics: {e}")
    
    def record_request_complete(
        self,
        context: Dict[str, Any],
        duration_seconds: Optional[float] = None,
        first_token_latency_seconds: Optional[float] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        success: bool = True,
        error_type: str = "",
        **extra_attributes
    ):
        """Record the completion of an LLM request (async, non-blocking).

        Args:
            context: Request context from record_request_start
            duration_seconds: Total request duration in seconds (will be converted to ms, calculated if not provided)
            first_token_latency_seconds: Time to first token/chunk in seconds (will be converted to ms)
            prompt_tokens: Number of prompt tokens (input tokens including cache)
            completion_tokens: Number of completion tokens (output tokens)
            total_tokens: Total tokens (calculated if not provided)
            cache_read_input_tokens: Number of tokens read from cache
            cache_creation_input_tokens: Number of tokens written to cache
            success: Whether the request succeeded
            error_type: Type of error if failed
            **extra_attributes: Additional attributes
        """
        if not self.enabled or not self.metrics_queue or not context:
            return

        start_time = context.get("start_time", time.time())
        attributes = context.get("attributes", {})

        # Calculate duration if not provided
        if duration_seconds is None:
            duration_seconds = time.time() - start_time

        # Convert seconds to milliseconds
        duration_ms = duration_seconds * 1000.0

        # Calculate total tokens if not provided
        if total_tokens == 0 and (prompt_tokens > 0 or completion_tokens > 0):
            total_tokens = prompt_tokens + completion_tokens

        # Default first_token_latency if not provided
        if first_token_latency_seconds is None:
            first_token_latency_seconds = 0.0

        # Convert seconds to milliseconds
        first_token_latency_ms = first_token_latency_seconds * 1000.0

        # Update attributes
        final_attributes = attributes.copy()
        final_attributes.update({
            "success": str(success).lower(),
            "error_type": error_type
        })
        final_attributes.update(extra_attributes)

        # Filter out empty values
        final_attributes = {k: v for k, v in final_attributes.items() if v}

        # Queue event asynchronously (non-blocking)
        event = RequestCompleteEvent(
            timestamp=time.time(),
            attributes=final_attributes,
            duration_ms=duration_ms,
            first_token_latency_ms=first_token_latency_ms,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=total_tokens,
            cache_read_tokens=cache_read_input_tokens,
            cache_write_tokens=cache_creation_input_tokens,
            success=success,
            error_type=error_type
        )
        self.metrics_queue.put_nowait(event)

        logger.debug("Queued LLM request completion")
    
    def _record_request_complete_sync(
        self,
        attributes: Dict[str, str],
        duration_ms: float,
        first_token_latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
        success: bool,
        error_type: str
    ):
        """Synchronously record request completion metrics (called by worker thread)."""
        try:
            # Record total duration (in milliseconds)
            self.llm_request_duration.record(duration_ms, attributes)

            # Record first token latency (TTFT) (in milliseconds)
            if first_token_latency_ms > 0:
                self.llm_first_token_latency.record(first_token_latency_ms, attributes)

            # Record detailed token usage by type
            # Counter: 记录所有类型的详细统计，使用API原始字段名
            if input_tokens > 0:
                token_attrs = attributes.copy()
                token_attrs["token_type"] = "prompt_tokens"
                self.llm_tokens_total.add(input_tokens, token_attrs)

            if output_tokens > 0:
                token_attrs = attributes.copy()
                token_attrs["token_type"] = "completion_tokens"
                self.llm_tokens_total.add(output_tokens, token_attrs)

            if cache_read_tokens > 0:
                token_attrs = attributes.copy()
                token_attrs["token_type"] = "cache_read_input_tokens"
                self.llm_tokens_total.add(cache_read_tokens, token_attrs)

            if cache_write_tokens > 0:
                token_attrs = attributes.copy()
                token_attrs["token_type"] = "cache_creation_input_tokens"
                self.llm_tokens_total.add(cache_write_tokens, token_attrs)

            if total_tokens > 0:
                token_attrs = attributes.copy()
                token_attrs["token_type"] = "total_tokens"
                self.llm_tokens_total.add(total_tokens, token_attrs)

            # Histogram: 只记录 total_tokens 的分布（避免指标爆炸）
            if total_tokens > 0:
                self.llm_tokens_per_request.record(total_tokens, attributes)

            # Record errors
            if not success:
                error_attrs = attributes.copy()
                self.llm_errors_total.add(1, error_attrs)

            # Decrement active requests (use original attributes without success/error_type)
            original_attrs = {k: v for k, v in attributes.items()
                            if k not in ["success", "error_type"]}
            self.llm_active_requests.add(-1, original_attrs)

        except Exception as e:
            logger.error(f"Failed to record request completion metrics: {e}")
    
    
    def _record_streaming_chunk_sync(
        self,
        attributes: Dict[str, str],
        chunk_size: int,
        chunk_tokens: int
    ):
        """Synchronously record streaming chunk metrics (called by worker thread)."""
        try:
            # Record chunk tokens if available
            if chunk_tokens > 0:
                token_attrs = attributes.copy()
                token_attrs["token_type"] = "chunk"
                self.llm_tokens_total.add(chunk_tokens, token_attrs)
                
        except Exception as e:
            logger.error(f"Failed to record streaming chunk metrics: {e}")
    
    def record_variant_routing(
        self,
        template_name: str,
        variant_name: str,
        route_name: str = "",
        routing_source: str = "normal",
        **extra_attributes
    ):
        """Record variant routing decision from conditional router pipeline for statistics (async).

        Args:
            template_name: Name of the prompt template
            variant_name: Name of the selected variant
            route_name: Name of the matched route
            routing_source: Source of routing decision
                - "normal": Regular routing (first-time or sticky disabled)
                - "sticky": Reused from sticky session cache
            **extra_attributes: Additional attributes for metrics
                - is_first_routing: "true" if first routing for this session+template+route, "false" otherwise
                  (only available when Redis is configured)
        """
        if not self.enabled:
            return

        try:
            # Queue event asynchronously (non-blocking)
            event = VariantRoutingEvent(
                timestamp=time.time(),
                attributes={},  # Not used in this event type
                template_name=template_name,
                variant_name=variant_name,
                route_name=route_name,
                routing_source=routing_source,
                extra_attributes=extra_attributes
            )
            self.metrics_queue.put_nowait(event)

            logger.debug(f"Queued variant routing: template={template_name}, "
                        f"variant={variant_name}, route={route_name}, source={routing_source}")

        except Exception as e:
            logger.error(f"Failed to queue variant routing: {e}")

    def _record_variant_routing_sync(
        self,
        template_name: str,
        variant_name: str,
        route_name: str = "",
        routing_source: str = "normal",
        **extra_attributes
    ):
        """Synchronously record variant routing decision (called by worker thread).

        Args:
            template_name: Name of the prompt template
            variant_name: Name of the selected variant
            route_name: Name of the matched route
            routing_source: Source of routing decision
            **extra_attributes: Additional attributes for metrics
        """
        try:
            attributes = {
                "template_name": template_name,
                "variant_name": variant_name,
                "route_name": route_name,
                "routing_source": routing_source,
                "pod_name": self.pod_name,
            }

            # Add extra attributes
            attributes.update(extra_attributes)

            # Filter out empty values
            attributes = {k: v for k, v in attributes.items() if v}

            # Record the routing decision
            self.variant_routing_total.add(1, attributes)

            logger.debug(f"Variant routing recorded: template={template_name}, "
                        f"variant={variant_name}, route={route_name}, source={routing_source}")

        except Exception as e:
            logger.error(f"Failed to record variant routing: {e}")

    def shutdown(self):
        """Shutdown metrics tracking and clean up resources."""
        if self.metrics_queue:
            self.metrics_queue.stop()
        logger.info("LLM metrics shutdown completed")


# Global instance
_llm_metrics: Optional[LLMMetrics] = None


def init_llm_metrics(
    service_name: str = "prompti",
    endpoint: str = "http://localhost:4317", 
    export_interval_ms: int = 5000,
    enabled: bool = True,
    queue_size: int = 10000
) -> LLMMetrics:
    """Initialize global LLM metrics instance.
    
    Args:
        service_name: Service name for metrics
        endpoint: OTEL collector endpoint
        export_interval_ms: Metrics export interval in milliseconds
        enabled: Whether metrics tracking is enabled
        queue_size: Max size of async metrics queue
        
    Returns:
        LLMMetrics instance
    """
    global _llm_metrics
    
    if _llm_metrics is None:
        _llm_metrics = LLMMetrics(
            service_name=service_name,
            endpoint=endpoint,
            export_interval_ms=export_interval_ms,
            enabled=enabled,
            queue_size=queue_size
        )
    
    return _llm_metrics


def get_llm_metrics() -> Optional[LLMMetrics]:
    """Get the global LLM metrics instance.
    
    Returns:
        LLMMetrics instance or None if not initialized
    """
    return _llm_metrics


def shutdown_llm_metrics():
    """Shutdown the global LLM metrics instance."""
    global _llm_metrics
    if _llm_metrics:
        _llm_metrics.shutdown()
        
def reset_llm_metrics():
    """Reset the global LLM metrics instance."""
    global _llm_metrics
    if _llm_metrics:
        _llm_metrics.shutdown()
    _llm_metrics = None