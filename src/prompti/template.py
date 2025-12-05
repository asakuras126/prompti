"""Prompt template with variant selection and Jinja rendering."""

from __future__ import annotations

import json
import re
import ast
from time import perf_counter
from typing import Any

from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment
from prometheus_client import Histogram
from pydantic import BaseModel, Field

from .model_client import ModelConfig

_env = SandboxedEnvironment(undefined=StrictUndefined)

_format_latency = Histogram(
    "prompt_format_latency_seconds",
    "Time spent formatting a prompt",
    labelnames=["template_name", "version"],
    registry=None,
)

SNAKE = re.compile(r"^[a-z][a-z0-9_]*$")


def _selector_to_flat(selector: dict[str, Any]) -> str:
    """Flatten selector to a lowercase JSON string for token matching."""
    return json.dumps(selector, separators=(",", ":")).lower()


def _parse_list_or_return_string(s: str):
    """
    Try to parse a string as a Python list.
    """
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return val
        else:
            # 解析成功但不是列表，返回原字符串
            return s
    except Exception:
        # 解析失败，返回原字符串
        return s


class Variant(BaseModel):
    """Single experiment arm."""

    selector: list[str] = []
    model_cfg: ModelConfig | None = Field(None)
    model_strategy: dict[str, Any] | None = Field(None)  # 高级模型策略配置
    messages: list[dict]
    required_variables: list[str] = []


class PromptTemplate(BaseModel):
    """Prompt template with multiple variants."""

    name: str
    description: str = ""
    version: str | None = None
    aliases: list[str] = []
    variants: dict[str, Variant]
    variant_router: dict[str, Any] | None = None
    id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptTemplate":
        """Create a PromptTemplate instance from a dictionary.
        
        This method handles the conversion of the template data dictionary
        (typically from the database) into a proper PromptTemplate instance.
        
        Args:
            data: Dictionary containing template data with the following structure:
                - name: Template name
                - description: Template description (optional)
                - version: Template version (optional)
                - template_id: Template ID (optional, mapped to 'id')
                - alias: List of aliases (optional, mapped to 'aliases')
                - variants: List of variant dictionaries containing:
                    - id: Variant name/key
                    - messages_template: List of message dictionaries
                    - required_variables: List of required variable names
                    - model_cfg: Model configuration dictionary
                    - tags: Additional tags/metadata
        
        Returns:
            PromptTemplate: A properly constructed PromptTemplate instance
            
        Example:
            >>> template_dict = {
            ...     "name": "customer-service",
            ...     "description": "Customer service template",
            ...     "version": "1.2.3",
            ...     "alias": ["latest", "stable"],
            ...     "variants": [
            ...         {
            ...             "id": "default",
            ...             "messages_template": [
            ...                 {"role": "system", "content": "You are a helpful assistant."},
            ...                 {"role": "user", "content": "{{user_query}}"}
            ...             ],
            ...             "required_variables": ["user_query"],
            ...             "model_cfg": {"model": "gpt-4o", "temperature": 0.1}
            ...         }
            ...     ]
            ... }
            >>> template = PromptTemplate.from_dict(template_dict)
        """
        # Extract basic fields
        name = data.get("name", "")
        description = data.get("description", "")
        version = data.get("version")
        template_id = data.get("template_id") or data.get("id")
        aliases = data.get("alias", []) or data.get("aliases", [])
        variant_router = data.get("variant_router")

        # Ensure aliases is a list
        if not isinstance(aliases, list):
            aliases = []

        # Process variants from list format to dict format
        variants_data = data.get("variants", [])
        variants = {}

        if isinstance(variants_data, list):
            for variant_data in variants_data:
                # Extract variant information
                variant_id = variant_data.get("id", "default")
                messages = variant_data.get("messages_template", [])
                required_variables = variant_data.get("required_variables", [])
                model_cfg_data = variant_data.get("model_cfg")
                model_strategy_data = variant_data.get("model_strategy")
                
                # 兼容性处理：如果model_strategy为空，但model_cfg中包含model_strategy，则从那里获取
                if model_strategy_data is None and model_cfg_data and isinstance(model_cfg_data, dict):
                    model_strategy_data = model_cfg_data.get("model_strategy")
                tags = variant_data.get("tags", {})

                # Create ModelConfig if present
                model_cfg = None
                if model_cfg_data and isinstance(model_cfg_data, dict):
                    # 过滤掉model_strategy字段，避免传递给ModelConfig
                    clean_model_cfg = {k: v for k, v in model_cfg_data.items() if k != "model_strategy"}
                    if clean_model_cfg:  # 只有在有其他配置时才创建ModelConfig
                        model_cfg = ModelConfig(**clean_model_cfg)

                # Create Variant instance
                variant = Variant(
                    selector=tags.get("selector", []) if isinstance(tags, dict) else [],
                    model_cfg=model_cfg,
                    model_strategy=model_strategy_data,
                    messages=messages,
                    required_variables=required_variables
                )

                variants[variant_id] = variant
        elif isinstance(variants_data, dict):
            # Handle case where variants is already a dict
            for variant_id, variant_data in variants_data.items():
                if isinstance(variant_data, dict):
                    messages = variant_data.get("messages", variant_data.get("messages_template", []))
                    required_variables = variant_data.get("required_variables", [])
                    model_cfg_data = variant_data.get("model_cfg")
                    model_strategy_data = variant_data.get("model_strategy")
                    
                    # 兼容性处理：如果model_strategy为空，但model_cfg中包含model_strategy，则从那里获取
                    if model_strategy_data is None and model_cfg_data and isinstance(model_cfg_data, dict):
                        model_strategy_data = model_cfg_data.get("model_strategy")
                        
                    selector = variant_data.get("selector", [])

                    # Create ModelConfig if present
                    model_cfg = None
                    if model_cfg_data and isinstance(model_cfg_data, dict):
                        # 过滤掉model_strategy字段，避免传递给ModelConfig
                        clean_model_cfg = {k: v for k, v in model_cfg_data.items() if k != "model_strategy"}
                        if clean_model_cfg:  # 只有在有其他配置时才创建ModelConfig
                            model_cfg = ModelConfig(**clean_model_cfg)

                    # Create Variant instance
                    variant = Variant(
                        selector=selector,
                        model_cfg=model_cfg,
                        model_strategy=model_strategy_data,
                        messages=messages,
                        required_variables=required_variables
                    )

                    variants[variant_id] = variant

        # If no variants found, create a default one
        if not variants:
            variants["default"] = Variant(
                selector=[],
                model_cfg=None,
                messages=[],
                required_variables=[]
            )

        # Create and return PromptTemplate instance
        return cls(
            name=name,
            description=description,
            version=version,
            aliases=aliases,
            variant_router=variant_router,
            variants=variants,
            id=template_id
        )

    def choose_variant(self, selector: dict[str, Any]) -> str | None:
        """Return the first variant id whose tokens all appear in ``selector``."""
        haystack = _selector_to_flat(selector)
        for vid, var in self.variants.items():
            if all(tok.lower() in haystack for tok in var.selector):
                return vid
        return None

    def format(
        self,
        variables: dict[str, Any],
        *,
        variant: str | None = None,
        selector: dict[str, Any] | None = None,
    ) -> tuple[list[dict], Variant]:
        """Render the template and return messages in OpenAI format."""
        start = perf_counter()
        try:
            selector = selector or variables
            variant = variant or self.choose_variant(selector) or next(iter(self.variants))
            """
            variants = {
                "prod_zh": Variant(selector=["prod", "zh-cn"]),
                "dev_en": Variant(selector=["dev", "en"]),
            }
            choose_variant({"env": "prod", "locale": "zh-CN"}) -> prod_zh
            
            """
            var = self.variants[variant]
            if not variables:
                return var.messages, var

            # Render messages with Jinja
            rendered_messages = []
            for msg in var.messages:
                role = msg.get("role")
                content = msg.get("content", [])
                tool_calls = msg.get("tool_calls", [])

                # Handle content rendering
                if isinstance(content, list):
                    rendered_content = []
                    for item in content:
                        if isinstance(item, dict):
                            item_type = item.get("type")
                            if item_type == "text":
                                text = item.get("text", "")
                                rendered = _env.from_string(text).render(**variables)
                                # 只有当渲染后的文本不为空时才添加
                                if rendered.strip():
                                    rendered_content.append({"type": "text", "text": rendered})
                            elif item_type == "image_url":
                                # Render image_url if it contains template variables
                                other_key = [k for k in item.keys() if k != "type"][0]
                                image_url = item.get(other_key, "")
                                rendered_url = _env.from_string(image_url).render(**variables)
                                parsed_rendered_url = _parse_list_or_return_string(rendered_url)
                                if isinstance(parsed_rendered_url, str):
                                    if parsed_rendered_url:
                                        rendered_content.append(
                                            {"type": "image_url", "image_url": {"url": parsed_rendered_url}})
                                else:
                                    for url in parsed_rendered_url:
                                        if url:
                                            rendered_content.append({"type": "image_url", "image_url": {"url": url}})
                            else:
                                # Pass through other content types
                                rendered_content.append(item)
                        else:
                            # Handle string content
                            text = str(item)
                            rendered = _env.from_string(text).render(**variables)
                            # 只有当渲染后的文本不为空时才添加
                            if rendered.strip():
                                rendered_content.append({"type": "text", "text": rendered})
                else:
                    # Handle single string content
                    text = str(content)
                    rendered = _env.from_string(text).render(**variables)
                    rendered_content = rendered

                # 只有当消息内容不为空时才添加到最终结果中
                # 对于列表类型的content，检查是否有有效内容
                # 对于字符串类型的content，检查是否为空
                should_add_message = False
                if isinstance(rendered_content, list) and len(rendered_content) > 0:
                    # 如果是列表且有有效内容项
                    should_add_message = True
                elif tool_calls:
                    should_add_message = True
                else:
                    # 如果是字符串且不为空
                    should_add_message = rendered_content and str(rendered_content).strip()
                if should_add_message:
                    rendered_message = {
                        "role": role,
                        "content": rendered_content
                    }
                    if tool_calls := msg.get("tool_calls", []):
                        rendered_message["tool_calls"] = tool_calls
                    if tool_call_id := msg.get("tool_call_id"):
                        rendered_message["tool_call_id"] = tool_call_id
                    rendered_messages.append(rendered_message)

            return rendered_messages, var
        finally:
            _format_latency.labels(self.name, self.version).observe(perf_counter() - start)
