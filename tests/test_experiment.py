"""Unit tests for ExperimentRegistry adapters and utilities."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from prompti.experiment import GrowthBookRegistry, UnleashRegistry, bucket
from prompti.model_client import ModelConfig
from prompti.template import PromptTemplate, Variant


def test_bucket_deterministic():
    split = {"A": 0.5, "B": 0.5}
    assert bucket("user1", split) == bucket("user1", split)
    assert bucket("user1", split) in {"A", "B"}


@pytest.mark.asyncio
async def test_unleash_registry():
    # Create a proper mock HTTP response
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {"name": "clarify", "variant": {"name": "A"}}
    mock_client.get.return_value = mock_response

    reg = UnleashRegistry("http://unleash", client=mock_client)
    split = await reg.aget_split("clarify", "u1")
    assert split.experiment_id == "clarify"
    assert split.variant == "A"

    # Verify the correct URL was called
    mock_client.get.assert_called_once()


@pytest.mark.asyncio
async def test_growthbook_registry():
    features = {"clarify": {"id": "clarify", "variants": {"A": 0.5, "B": 0.5}}}
    reg = GrowthBookRegistry(features)
    split = await reg.aget_split("clarify", "user1")
    assert split.traffic_split == {"A": 0.5, "B": 0.5}
    assert split.variant is None


def test_choose_variant():
    tmpl = PromptTemplate(
        name="demo",
        description="",
        version="1",
        variants={
            "a": Variant(selector=["vip"], model_config=ModelConfig(provider="x", model="m1"), messages=[]),
            "b": Variant(selector=["guest"], model_config=ModelConfig(provider="x", model="m1"), messages=[]),
        },
    )
    ctx = {"role": "vip-user"}
    assert tmpl.choose_variant(ctx) == "a"
