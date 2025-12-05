"""OpenAI format message types used throughout the package."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class Message(BaseModel):
    """OpenAI format message for input/output.
    
    This is the standard message format used by OpenAI and LiteLLM,
    supporting text content, tool calls, and tool results.
    For multimodal messages (vision), content can be a list of objects.
    """

    role: str = Field(..., description="The role of the message sender")
    content: Optional[Union[str, List[Dict[str, Any]]]] = Field(
        None,
        description="The content of the message. Can be a string for text-only messages, or a list of content objects for multimodal messages (e.g., text + images)"
    )
    reasoning_content: Optional[str] = Field(None, description="The content of the message for reasoning")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls made by the assistant")
    tool_call_id: Optional[str] = Field(None, description="ID of the tool call this message is responding to")

    def to_openai(self) -> Dict[str, Any]:
        """Convert to OpenAI format dictionary."""
        result = {"role": self.role}
        if self.content is not None:
            result["content"] = self.content
        if self.tool_calls is not None:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id
        return result

    @classmethod
    def from_openai(cls, data: Dict[str, Any]) -> 'Message':
        """Create from OpenAI format dictionary."""
        return cls(
            role=data.get("role", "user"),
            content=data.get("content"),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id")
        )

    @classmethod
    def get_openai_messages(cls, messages: List[Dict[str, Any]]) -> List['Message']:
        """Convert list of OpenAI format message dictionaries to list of Message objects.
        
        Args:
            messages: List of OpenAI format message dictionaries
            
        Returns:
            List of Message objects
        """
        return [cls.from_openai(msg) for msg in messages]

    @classmethod
    def create_user(cls, content: Union[str, List[Dict[str, Any]]]) -> 'Message':
        """Create a user message with text or multimodal content."""
        return cls(role="user", content=content)

    @classmethod
    def create_user_text(cls, text: str) -> 'Message':
        """Create a user message with text content."""
        return cls(role="user", content=text)

    @classmethod
    def create_user_multimodal(cls, content_objects: List[Dict[str, Any]]) -> 'Message':
        """Create a user message with multimodal content (text + images, etc.)."""
        return cls(role="user", content=content_objects)

    @classmethod
    def create_user_with_image(cls, text: str, image_url: str, detail: str = "auto") -> 'Message':
        """Create a user message with text and image.

        Args:
            text: The text content
            image_url: URL or base64 data URL of the image
            detail: Image detail level ("low", "high", or "auto")
        """
        content = [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": detail
                }
            }
        ]
        return cls(role="user", content=content)

    @classmethod
    def create_assistant(cls, content: str) -> 'Message':
        """Create an assistant message."""
        return cls(role="assistant", content=content)

    @classmethod
    def create_system(cls, content: str) -> 'Message':
        """Create a system message."""
        return cls(role="system", content=content)

    @classmethod
    def create_tool_result(cls, content: str, tool_call_id: str) -> 'Message':
        """Create a tool result message."""
        return cls(role="tool", content=content, tool_call_id=tool_call_id)

    @classmethod
    def create_tool_call(cls, tool_calls: List[Dict[str, Any]]) -> 'Message':
        """Create a tool call message."""
        return cls(role="assistant", content=None, tool_calls=tool_calls)

    def has_tool_calls(self) -> bool:
        """Check if this message contains tool calls."""
        return bool(self.tool_calls)

    def get_tool_call_names(self) -> List[str]:
        """Get list of tool call function names."""
        if not self.tool_calls:
            return []
        return [tool_call.get("function", {}).get("name", "") for tool_call in self.tool_calls]

    def get_tool_call_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get the first tool call with the specified function name."""
        if not self.tool_calls:
            return None
        for tool_call in self.tool_calls:
            if tool_call.get("function", {}).get("name") == name:
                return tool_call
        return None


class Usage(BaseModel):
    """Token usage information following OpenAI format."""

    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens used")
    
    # Cache-related fields for providers that support caching (e.g., Anthropic)
    cache_creation_input_tokens: Optional[int] = Field(None, description="Number of tokens used to create cache")
    cache_read_input_tokens: Optional[int] = Field(None, description="Number of tokens read from cache")
    cached_tokens: Optional[int] = Field(None, description="Number of tokens that were cached")


class Choice(BaseModel):
    """A single choice from the model response following OpenAI format."""

    index: int = Field(..., description="Index of the choice")
    message: Message = Field(..., description="The message content")
    finish_reason: Optional[str] = Field(None, description="Reason for finishing the response")
    logprobs: Optional[Dict[str, Any]] = Field(None, description="Log probabilities for the choice")


class ModelResponse(BaseModel):
    """Complete model response including metadata following OpenAI format.
    
    This represents the full response from a language model, including
    the generated content, metadata, and usage information.
    """

    id: Optional[str] = Field(None, description="Unique identifier for the response")
    object: Optional[str] = Field(None, description="Object type")
    created: Optional[int] = Field(None, description="Unix timestamp of creation")
    model: Optional[str] = Field(None, description="Model name used for generation")
    choices: Optional[List[Choice]] = Field(None, description="List of generated choices")
    usage: Optional[Usage] = Field(None, description="Token usage information")

    # Additional fields that may be present
    system_fingerprint: Optional[str] = Field(None, description="System fingerprint")
    error: Optional[Dict[str, Any]] = Field(None, description="Error object if the request failed")

    def get_content(self) -> Optional[Union[str, List[Dict[str, Any]]]]:
        """Get the content from the first choice."""
        if self.choices and self.choices[0].message.content:
            return self.choices[0].message.content
        return None

    def get_text_content(self) -> Optional[str]:
        """Get text content from the first choice.
        If content is a list, extract text from text-type objects."""
        if not self.choices or self.choices[0].message.content is None:
            return None

        content = self.choices[0].message.content
        if isinstance(content, str):
            return content  # 返回空字符串或实际内容
        elif isinstance(content, list):
            # Extract text from content objects
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return "\n".join(text_parts) if text_parts else None
        return None

    def get_tool_calls(self) -> Optional[List[Dict[str, Any]]]:
        """Get tool calls from the first choice."""
        if self.choices and self.choices[0].message.tool_calls:
            return self.choices[0].message.tool_calls
        return None

    def get_finish_reason(self) -> Optional[str]:
        """Get finish reason from the first choice."""
        if self.choices:
            return self.choices[0].finish_reason
        return None

    def get_message(self) -> Optional[Message]:
        """Get the message from the first choice."""
        if self.choices:
            return self.choices[0].message
        return None

    prompt_filter_results: Optional[List[Dict[str, Any]]] = Field(None, description="Prompt filter results")


class StreamingChoice(BaseModel):
    """A single choice from a streaming response following OpenAI format."""

    index: int = Field(..., description="Index of the choice")
    delta: Message = Field(..., description="The delta message content")
    finish_reason: Optional[str] = Field(None, description="Reason for finishing the response")
    logprobs: Optional[Dict[str, Any]] = Field(None, description="Log probabilities for the choice")


class StreamingModelResponse(BaseModel):
    """Streaming model response following OpenAI format.
    
    This represents a chunk of a streaming response from a language model,
    containing partial content and metadata.
    """

    id: Optional[str] = Field(None, description="Unique identifier for the response")
    object: Optional[str] = Field(None, description="Object type")
    created: Optional[int] = Field(None, description="Unix timestamp of creation")
    model: Optional[str] = Field(None, description="Model name used for generation")
    choices: Optional[List[StreamingChoice]] = Field(None, description="List of streaming choices")
    usage: Optional[Usage] = Field(None, description="Token usage information")

    # Additional fields that may be present
    system_fingerprint: Optional[str] = Field(None, description="System fingerprint")
    error: Optional[Dict[str, Any]] = Field(None, description="Error object if the request failed")

    def get_content(self) -> Optional[Union[str, List[Dict[str, Any]]]]:
        """Get the content from the first choice delta."""
        if self.choices and self.choices[0].delta.content:
            return self.choices[0].delta.content
        return None

    def get_text_content(self) -> Optional[str]:
        """Get text content from the first choice delta.
        If content is a list, extract text from text-type objects."""
        if not self.choices or self.choices[0].delta.content is None:
            return None

        content = self.choices[0].delta.content
        if isinstance(content, str):
            return content  # 返回空字符串或实际内容
        elif isinstance(content, list):
            # Extract text from content objects
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return "\n".join(text_parts) if text_parts else None
        return None

    def get_tool_calls(self) -> Optional[List[Dict[str, Any]]]:
        """Get tool calls from the first choice delta."""
        if self.choices and self.choices[0].delta.tool_calls:
            return self.choices[0].delta.tool_calls
        return None

    def get_finish_reason(self) -> Optional[str]:
        """Get finish reason from the first choice."""
        if self.choices:
            return self.choices[0].finish_reason
        return None

    def get_delta(self) -> Optional[Message]:
        """Get the delta message from the first choice."""
        if self.choices:
            return self.choices[0].delta
        return None


# 为了向后兼容，保留原有的 Message 类作为主要接口
__all__ = [
    "Message",
    "Usage",
    "Choice",
    "ModelResponse",
    "StreamingChoice",
    "StreamingModelResponse"
]
