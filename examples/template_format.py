"""Format a template with :class:`PromptEngine` and send via LiteLLM."""

import asyncio
import os

from prompti.engine import PromptEngine, Setting


async def main() -> None:
    """Render ``support_reply`` template and send via ``litellm``."""
    try:
        import litellm
    except ImportError as e:
        raise ImportError("litellm is required for this example. Install with: pip install 'prompti[litellm]'") from e
    engine = PromptEngine.from_setting(Setting.from_file("examples/configs/settings.yaml"))

    # Format the template directly as OpenAI messages and send
    messages = await engine.format(
        "support_reply",
        {"name": "Ada", "issue": "login failed"}
    )
    print(messages)
    response = await litellm.acompletion(
        model=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
        messages=messages,
        api_key=os.getenv("LITELLM_API_KEY"),
        base_url=os.getenv("LITELLM_ENDPOINT"),
    )

    print(response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    asyncio.run(main())
