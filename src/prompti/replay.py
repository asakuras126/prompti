"""Replay and recording utilities."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Callable, Iterable
from typing import Union
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import aiofiles
from prometheus_client import Counter

from .message import Message, ModelResponse, StreamingModelResponse
from .model_client import ModelClient, ModelConfig, RunParams


class ReplayError(Exception):
    """Raised when replay encounters an error event."""


_replay_counter = Counter("trace_replay_total", "replay summary", labelnames=["status"])
_diff_tokens = Counter("trace_diff_tokens_total", "token diff")


class ModelClientRecorder(ModelClient):
    """Wrapper around :class:`ModelClient` that records all I/O."""

    def __init__(
        self,
        client: ModelClient,
        session_id: str,
        output_dir: str | Path | None = None,
    ) -> None:
        """Wrap ``client`` and log all interactions under ``session_id``."""
        super().__init__(client.cfg, client=client._client)
        self._wrapped = client
        self.session_id = session_id
        self.output_dir = Path(output_dir or Path.home() / ".prompt/sessions")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def _write_row(self, file, step: int, direction: str, payload: dict | list, meta: dict) -> None:
        row = {
            "session_id": self.session_id,
            "trace_id": self._trace_id,
            "step": step,
            "direction": direction,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
            "meta": meta,
        }
        await file.write(json.dumps(row, ensure_ascii=False) + "\n")

    async def _run(
        self,
        params: RunParams,
    ) -> AsyncGenerator[Union[ModelResponse, StreamingModelResponse], None]:
        self._trace_id = str(uuid4())
        step = 0
        meta = {"provider": self.cfg.provider, "model": self.cfg.model}
        log_file = self.output_dir / f"rollout-{datetime.now(timezone.utc).date().isoformat()}-{self.session_id}.jsonl"
        async with aiofiles.open(log_file, "a") as f:
            await self._write_row(
                f,
                step,
                "req",
                [m.model_dump() for m in params.messages],
                meta,
            )
            step += 1
            try:
                async for response in self._wrapped.arun(params):
                    # Determine direction based on response type
                    direction = "delta" if isinstance(response, StreamingModelResponse) else "res"
                    await self._write_row(f, step, direction, response.model_dump(), meta)
                    step += 1
                    yield response
            except Exception as exc:
                await self._write_row(f, step, "error", {"error": str(exc)}, meta)
                raise

    async def aclose(self) -> None:
        """Close the underlying client."""
        await self._wrapped.aclose()


class ReplayEngine:
    """Replays a recorded trace."""

    def __init__(self, client_factory: Callable[[str], ModelClient]):
        """Create with a ``client_factory`` mapping provider -> ModelClient."""
        self._client_factory = client_factory
        self._clients: dict[str, ModelClient] = {}

    def _get_client(self, provider: str) -> ModelClient:
        if provider not in self._clients:
            self._clients[provider] = self._client_factory(provider)
        return self._clients[provider]

    async def areplay(
        self,
        rows: Iterable[dict],
        up_to_step: int | None = None,
        patch: dict[int, list[Message]] | None = None,
    ) -> AsyncGenerator[Union[ModelResponse, StreamingModelResponse], None]:
        """Replay a recorded session and optionally patch messages."""
        patch = patch or {}
        status = "ok"
        try:
            for row in sorted(rows, key=lambda r: r["step"]):
                if up_to_step is not None and row["step"] > up_to_step:
                    break
                direction = row["direction"]
                if direction == "req":
                    step = row["step"]
                    msgs = patch.get(step)
                    if msgs is None:
                        msgs = [Message(**m) for m in row["payload"]]
                    meta = row.get("meta", {})
                    provider = meta.get("provider", "")
                    model = meta.get("model", "")
                    client = self._get_client(provider)
                    cfg = ModelConfig(provider=provider, model=model)
                    client.cfg = cfg  # update static config if different
                    params = RunParams(messages=msgs)
                    async for response in client.arun(params):
                        yield response
                elif direction in ("delta", "res", "tool_result"):
                    # Handle both new format (full response) and old format (just message) for backward compatibility
                    payload = row["payload"]

                    # Check if payload is already a complete response (new format)
                    if "choices" in payload:
                        # New format: payload is already a ModelResponse or StreamingModelResponse
                        if direction == "delta":
                            yield StreamingModelResponse(**payload)
                        else:
                            yield ModelResponse(**payload)
                    else:
                        # Old format: payload is just a Message dict, reconstruct response
                        if direction == "delta":
                            # Create StreamingResponse from stored data
                            yield StreamingModelResponse(
                                id=payload.get("id", "replay"),
                                model=model,
                                choices=[{
                                    "index": 0,
                                    "delta": Message(**payload),
                                    "finish_reason": payload.get("finish_reason")
                                }]
                            )
                        else:
                            # Create ModelResponse from stored data
                            yield ModelResponse(
                                id=payload.get("id", "replay"),
                                model=model,
                                choices=[{
                                    "index": 0,
                                    "message": Message(**payload),
                                    "finish_reason": payload.get("finish_reason", "stop")
                                }]
                            )
                elif direction == "error":
                    status = "fail"
                    raise ReplayError(row.get("payload"))
        finally:
            _replay_counter.labels(status).inc()
