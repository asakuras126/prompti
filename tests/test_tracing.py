import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from prompti.message import Message
from prompti.model_client import ModelClient, ModelConfig, RunParams


class DummyClient(ModelClient):
    provider = "dummy"
    async def _run(self, params: RunParams):
        yield Message(role="assistant", kind="text", content="ok")

@pytest.mark.asyncio
async def test_span_has_request_attrs():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    client = DummyClient(ModelConfig(provider="dummy", model="x"))
    params = RunParams(
        messages=[Message(role="user", kind="text", content="hi")],
        request_id="r1",
        session_id="s1",
        user_id="u1",
    )

    out = [m async for m in client.arun(params)]
    assert out[0].content == "ok"

    spans = exporter.get_finished_spans()
    assert spans
    attrs = spans[0].attributes
    assert attrs["http.request_id"] == "r1"
    assert attrs["user.session_id"] == "s1"
    assert attrs["user.id"] == "u1"
