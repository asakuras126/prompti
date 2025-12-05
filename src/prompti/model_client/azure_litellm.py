"""Azure LiteLLM client implementation specifically for Azure OpenAI models."""

from __future__ import annotations

import os
import json
import time
import uuid
from typing import Any, Dict, List, Union
from collections.abc import AsyncGenerator, Generator

from .base import ModelClient, SyncModelClient, ModelConfig, RunParams, should_retry_error, calculate_retry_delay, handle_model_client_error
from ..message import Message, ModelResponse, StreamingModelResponse, Usage, Choice, StreamingChoice
from ..logger import get_logger
import litellm
import asyncio

litellm.drop_params = True

logger = get_logger(__name__)


def messages_to_responses_items(messages: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], str]:
    """Convert Chat Completions messages into Responses API input items.
    
    Returns:
        tuple: (responses_items, instructions) where instructions contains system message content
    """
    if not messages:
        return [], ''

    out: List[Dict[str, Any]] = []
    instructions = ''
    
    for m in messages:
        msg = dict(m)
        role = str(msg.get('role', ''))
        content = msg.get('content', '')
        
        # Extract system messages as instructions
        if role == 'system':
            if isinstance(content, str):
                instructions = content
            elif isinstance(content, list):
                # Extract text from list content
                text_parts = []
                for seg in content:
                    if isinstance(seg, dict) and seg.get('type') == 'text' and seg.get('text'):
                        text_parts.append(seg['text'])
                if text_parts:
                    instructions = ' '.join(text_parts)
            continue
        
        if role == 'tool':
            # Tool messages become function_call_output items
            call_id = str(msg.get('tool_call_id', ''))
            output_text = ''
            if isinstance(content, str):
                output_text = content
            elif isinstance(content, list):
                for seg in content:
                    if isinstance(seg, dict) and seg.get('type') == 'text' and seg.get('text'):
                        output_text = seg['text']
                        break
            out.append({'type': 'function_call_output', 'call_id': call_id, 'output': output_text})
            continue

        if role == 'assistant':
            # Assistant messages with content
            if content:
                if isinstance(content, str) and content:
                    out.append({'role': 'assistant', 'content': content})
                elif isinstance(content, list):
                    # Extract text from list content
                    text_parts = []
                    for seg in content:
                        if isinstance(seg, dict) and seg.get('type') == 'text' and seg.get('text'):
                            text_parts.append(seg['text'])
                    if text_parts:
                        out.append({'role': 'assistant', 'content': ' '.join(text_parts)})
            
            # Assistant tool calls become function_call items
            tool_calls = msg.get('tool_calls')
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get('function', {})
                    out.append({
                        'type': 'function_call',
                        'call_id': str(tc.get('id', '')),
                        'name': str(fn.get('name', '')),
                        'arguments': str(fn.get('arguments', ''))
                    })
            continue

        if role in {'user', 'developer'}:
            # User/developer messages - keep multimodal content as-is
            if isinstance(content, str):
                if content:
                    out.append({'role': role, 'content': content})
            elif isinstance(content, list):
                # For multimodal content, convert to the message format with structured content
                # Responses API expects messages to have structured content, not separate items
                message_content = []
                for seg in content:
                    if isinstance(seg, dict):
                        if seg.get('type') == 'text' and seg.get('text'):
                            message_content.append({'type': 'input_text', 'text': seg['text']})
                        elif seg.get('type') == 'image_url' and seg.get('image_url'):
                            image_url_obj = seg['image_url']
                            url = image_url_obj.get('url') if isinstance(image_url_obj, dict) else image_url_obj
                            if url:
                                message_content.append({'type': 'input_image', 'image_url': url})
                
                if message_content:
                    # Create a single message item with structured content
                    out.append({'role': role, 'content': message_content})
            elif content is not None:
                out.append({'role': role, 'content': str(content)})
            continue

        raise ValueError(f'Unsupported message role: {role}')
    return out, instructions


def responses_to_completion_format(responses_result: Any) -> Dict[str, Any]:
    """Convert Responses API result to ChatCompletions format."""
    output_items = responses_result.output if hasattr(responses_result, 'output') else []
    
    logger.debug(f"Response output_items: {output_items}")
    logger.debug(f"Response output_items type: {type(output_items)}")

    content = ''
    reasoning_content = ''
    tool_calls: List[Dict[str, Any]] = []

    for item in output_items:
        # Always use getattr first for objects, only fall back to dict access
        item_type = getattr(item, 'type', None)
        logger.debug(f"Processing item type: {item_type}, item: {item}")
        
        if item_type == 'message':
            item_content = getattr(item, 'content', None)
            logger.debug(f"Message content: {item_content}, type: {type(item_content)}")
            if item_content:
                if isinstance(item_content, list):
                    for seg in item_content:
                        seg_type = getattr(seg, 'type', None)
                        logger.debug(f"Segment type: {seg_type}, seg: {seg}")
                        # Check for both 'text' and 'output_text' types
                        if seg_type in ('text', 'output_text'):
                            text = getattr(seg, 'text', None)
                            logger.debug(f"Extracted text: {text}")
                            if text:
                                content = text
                elif isinstance(item_content, str):
                    content = item_content
                    logger.debug(f"Direct string content: {content}")
            continue
            
        if item_type == 'function_call':
            call_id = getattr(item, 'call_id', None) or getattr(item, 'id', None) or ''
            name = getattr(item, 'name', None) or ''
            arguments = getattr(item, 'arguments', None) or ''
            tool_calls.append({'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': arguments}})
            continue
            
        if item_type == 'reasoning':
            # Handle reasoning item - it's an object, not a dict
            item_content = getattr(item, 'content', None)
            if item_content and isinstance(item_content, list):
                parts = []
                for seg in item_content:
                    text = getattr(seg, 'text', None)
                    if text:
                        parts.append(text)
                if parts:
                    reasoning_content = '\n\n'.join(parts)
            # Check for summary attribute
            if not reasoning_content and hasattr(item, 'summary'):
                summary = getattr(item, 'summary', None)
                if summary and isinstance(summary, list):
                    parts = []
                    for s in summary:
                        text = getattr(s, 'text', None)
                        if text:
                            parts.append(text)
                    if parts:
                        reasoning_content = '\n\n'.join(parts)

    # Build message - include content even when there are tool_calls
    message: Dict[str, Any] = {'role': 'assistant'}
    
    # Always include content if present (even with tool_calls)
    if content:
        message['content'] = content
    else:
        # If no content and there are tool_calls, set content to None (OpenAI standard)
        message['content'] = None if tool_calls else ''
    
    if tool_calls:
        message['tool_calls'] = tool_calls
    if reasoning_content:
        message['reasoning_content'] = reasoning_content

    model = getattr(responses_result, 'model', '')
    finish_reason = 'tool_calls' if tool_calls else 'stop'

    response = {
        'id': getattr(responses_result, 'id', ''),
        'object': 'chat.completion',
        'created': int(getattr(responses_result, 'created_at', None) or getattr(responses_result, 'created', None) or 0),
        'model': model,
        'choices': [{'index': 0, 'message': message, 'finish_reason': finish_reason}],
        'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    }

    usage = getattr(responses_result, 'usage', None)
    if usage is not None:
        input_tokens = getattr(usage, 'input_tokens', 0)
        output_tokens = getattr(usage, 'output_tokens', 0)
        total_tokens = getattr(usage, 'total_tokens', input_tokens + output_tokens)
        
        response['usage']['prompt_tokens'] = input_tokens
        response['usage']['completion_tokens'] = output_tokens
        response['usage']['total_tokens'] = total_tokens

        output_details = getattr(usage, 'output_tokens_details', None)
        if output_details:
            rt = getattr(output_details, 'reasoning_tokens', None)
            if isinstance(rt, int):
                response['usage']['completion_tokens_details'] = {'reasoning_tokens': rt}

    return response


class AzureLiteLLMClient(ModelClient):
    """Azure-specific LiteLLM client for Azure OpenAI models with special handling."""

    provider = "azure_litellm"

    def __init__(self, cfg: ModelConfig, client=None, is_debug: bool = False) -> None:
        """Initialize Azure LiteLLM client with Azure-specific configurations."""
        super().__init__(cfg, client, is_debug=is_debug)
        self.api_key = cfg.api_key
        self.api_url = cfg.api_url
        self._setup_azure_environment()

    def _setup_azure_environment(self):
        """Setup Azure-specific environment and configurations."""
        if not hasattr(litellm, 'model_cost'):
            litellm.model_cost = {}
        
        model_name = self.cfg.get_actual_model_name()
        litellm.model_cost[model_name] = {"mode": "responses"}
        logger.info(f"Configured {model_name} to use Response API Bridge")

        if self.api_key and 'AZURE_API_KEY' not in os.environ:
            os.environ['AZURE_API_KEY'] = self.api_key
        if self.api_url and 'AZURE_API_BASE' not in os.environ:
            os.environ['AZURE_API_BASE'] = self.api_url
        if 'AZURE_API_VERSION' not in os.environ:
            os.environ['AZURE_API_VERSION'] = '2024-10-01-preview'

    async def _run(self, params: RunParams) -> AsyncGenerator[Union[ModelResponse, StreamingModelResponse], None]:
        """Execute call using litellm.responses() API with format conversion."""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1}/{max_retries}")
                
                messages_dict = [m.to_openai() for m in params.messages]
                responses_items, instructions = messages_to_responses_items(messages_dict)
                
                request_data = self._build_responses_request_data(params, responses_items, instructions)
                
                if params.stream:
                    response_stream = await litellm.aresponses(**request_data)
                    async for chunk in response_stream:
                        yield self._convert_stream_chunk(chunk)
                    return
                else:
                    responses_result = await litellm.aresponses(**request_data)
                    logger.debug(f"Raw responses_result: {responses_result}")
                    logger.debug(f"responses_result type: {type(responses_result)}")
                    
                    completion_dict = responses_to_completion_format(responses_result)
                    yield self._dict_to_model_response(completion_dict)
                    return
                    
            except Exception as e:
                is_last_attempt = attempt == max_retries - 1
                should_retry = should_retry_error(e)
                
                if should_retry and not is_last_attempt:
                    wait_time = calculate_retry_delay(attempt, error=e)
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Request failed permanently: {str(e)}")
                    yield handle_model_client_error(e, params.stream, self._create_error_response)
                    return

    def _convert_stream_chunk(self, chunk: Any) -> StreamingModelResponse:
        """Convert streaming chunk to StreamingModelResponse."""
        delta_content = None
        tool_calls = None
        finish_reason = None
        
        if hasattr(chunk, 'choices') and chunk.choices:
            choice = chunk.choices[0]
            if hasattr(choice, 'delta'):
                delta = choice.delta
                delta_content = getattr(delta, 'content', None)
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    tool_calls = []
                    for tc in delta.tool_calls:
                        tool_calls.append({
                            'id': getattr(tc, 'id', ''),
                            'type': 'function',
                            'function': {
                                'name': getattr(tc.function, 'name', '') if hasattr(tc, 'function') else '',
                                'arguments': getattr(tc.function, 'arguments', '') if hasattr(tc, 'function') else ''
                            }
                        })
            finish_reason = getattr(choice, 'finish_reason', None)
        
        usage = None
        if hasattr(chunk, 'usage') and chunk.usage:
            usage = Usage(
                prompt_tokens=getattr(chunk.usage, 'prompt_tokens', 0),
                completion_tokens=getattr(chunk.usage, 'completion_tokens', 0),
                total_tokens=getattr(chunk.usage, 'total_tokens', 0)
            )
        
        return StreamingModelResponse(
            id=getattr(chunk, 'id', str(uuid.uuid4())),
            created=getattr(chunk, 'created', int(time.time())),
            model=getattr(chunk, 'model', self.cfg.get_aggregated_model_name()),
            choices=[StreamingChoice(
                index=0,
                delta=Message(role='assistant', content=delta_content, tool_calls=tool_calls),
                finish_reason=finish_reason
            )],
            usage=usage
        )

    def _dict_to_model_response(self, completion_dict: Dict[str, Any]) -> ModelResponse:
        """Convert completion dict to ModelResponse."""
        choice_data = completion_dict['choices'][0]
        message_data = choice_data['message']
        
        usage = None
        if 'usage' in completion_dict:
            usage_data = completion_dict['usage']
            usage = Usage(
                prompt_tokens=usage_data.get('prompt_tokens', 0),
                completion_tokens=usage_data.get('completion_tokens', 0),
                total_tokens=usage_data.get('total_tokens', 0)
            )
        
        return ModelResponse(
            id=completion_dict.get('id', str(uuid.uuid4())),
            created=completion_dict.get('created', int(time.time())),
            model=completion_dict.get('model', self.cfg.get_aggregated_model_name()),
            choices=[Choice(
                index=0,
                message=Message(
                    role=message_data['role'],
                    content=message_data.get('content'),
                    tool_calls=message_data.get('tool_calls')
                ),
                finish_reason=choice_data.get('finish_reason', 'stop')
            )],
            usage=usage
        )

    def _fix_tool_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively fix tool schema issues for Azure OpenAI compatibility."""
        if not isinstance(schema, dict):
            return schema
        
        fixed_schema = {}
        for key, value in schema.items():
            if isinstance(value, dict):
                fixed_value = self._fix_tool_schema(value)
                if fixed_value.get("type") == "array" and "items" not in fixed_value:
                    fixed_value["items"] = {"type": "object"}
                fixed_schema[key] = fixed_value
            elif isinstance(value, list):
                fixed_schema[key] = [self._fix_tool_schema(item) if isinstance(item, dict) else item for item in value]
            else:
                fixed_schema[key] = value
        return fixed_schema

    def _build_responses_request_data(self, params: RunParams, responses_items: List[Dict[str, Any]], instructions: str = '') -> Dict[str, Any]:
        """Build request data for litellm.responses() API."""
        model_name = self.cfg.get_actual_model_name()
        
        request_data = {
            'model': model_name,
            'input': responses_items,
            'stream': params.stream,
            'parallel_tool_calls': False  # Disable parallel tool calls
        }
        
        if params.timeout is not None:
            request_data['timeout'] = params.timeout
            request_data['request_timeout'] = params.timeout
        
        request_data['num_retries'] = 0
        
        if instructions:
            request_data['instructions'] = instructions
        
        if params.temperature is not None:
            request_data['temperature'] = params.temperature
        elif self.cfg.temperature is not None:
            request_data['temperature'] = self.cfg.temperature
        
        if params.max_tokens is not None:
            request_data['max_tokens'] = params.max_tokens
        elif self.cfg.max_tokens is not None:
            request_data['max_tokens'] = self.cfg.max_tokens
        
        # Add tools if present - Responses API uses flat format without 'function' wrapper
        if params.tool_params and params.tool_params.tools:
            tools = []
            for tool in params.tool_params.tools:
                if hasattr(tool, 'name'):
                    tools.append({
                        'type': 'function',
                        'name': tool.name,
                        'description': tool.description,
                        'parameters': self._fix_tool_schema(tool.parameters)
                    })
                elif isinstance(tool, dict) and 'function' in tool:
                    fn = tool['function']
                    tools.append({
                        'type': 'function',
                        'name': fn['name'],
                        'description': fn['description'],
                        'parameters': self._fix_tool_schema(fn['parameters'])
                    })
            request_data['tools'] = tools
            
            # Require the model to use a tool when tools are provided
            # Use 'required' instead of {"type": "function"} for Azure Responses API
            request_data['tool_choice'] = 'required'
        
        logger.debug(f"Responses API request: {json.dumps(request_data, indent=2, ensure_ascii=False)}")
        
        # Save request data to trace context for monitoring/debugging
        params.trace_context["llm_request_body"] = request_data
        
        return request_data

    def _create_error_response(self, error_message: str, is_streaming: bool = False) -> Union[ModelResponse, StreamingModelResponse]:
        """Create error response."""
        error_object = {"message": error_message, "type": "api_error", "code": "azure_error"}
        if is_streaming:
            return StreamingModelResponse(error=error_object)
        else:
            return ModelResponse(error=error_object)


class SyncAzureLiteLLMClient(SyncModelClient):
    """Synchronous Azure-specific LiteLLM client for Azure OpenAI models."""

    provider = "azure_litellm"

    def __init__(self, cfg: ModelConfig, client=None, is_debug: bool = False) -> None:
        """Initialize synchronous Azure LiteLLM client with Azure-specific configurations."""
        super().__init__(cfg, client, is_debug=is_debug)
        self.api_key = cfg.api_key
        self.api_url = cfg.api_url
        self._setup_azure_environment()

    def _setup_azure_environment(self):
        """Setup Azure-specific environment and configurations."""
        if not hasattr(litellm, 'model_cost'):
            litellm.model_cost = {}
        
        model_name = self.cfg.get_actual_model_name()
        litellm.model_cost[model_name] = {"mode": "responses"}
        logger.info(f"Configured {model_name} to use Response API Bridge")

        if self.api_key and 'AZURE_API_KEY' not in os.environ:
            os.environ['AZURE_API_KEY'] = self.api_key
        if self.api_url and 'AZURE_API_BASE' not in os.environ:
            os.environ['AZURE_API_BASE'] = self.api_url
        if 'AZURE_API_VERSION' not in os.environ:
            os.environ['AZURE_API_VERSION'] = '2024-10-01-preview'

    def _run(self, params: RunParams) -> Generator[Union[ModelResponse, StreamingModelResponse], None, None]:
        """Execute call using litellm.responses() API with format conversion."""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1}/{max_retries}")
                
                messages_dict = [m.to_openai() for m in params.messages]
                responses_items, instructions = messages_to_responses_items(messages_dict)
                
                request_data = self._build_responses_request_data(params, responses_items, instructions)
                
                if params.stream:
                    response_stream = litellm.responses(**request_data)
                    for chunk in response_stream:
                        yield self._convert_stream_chunk(chunk)
                    return
                else:
                    responses_result = litellm.responses(**request_data)
                    logger.debug(f"responses_result type: {type(responses_result)}")
                    logger.debug(f"responses_result has output: {hasattr(responses_result, 'output')}")
                    logger.info(f"Raw responses output result: {responses_result.output if hasattr(responses_result, 'output') else None}")
                    
                    completion_dict = responses_to_completion_format(responses_result)
                    yield self._dict_to_model_response(completion_dict)
                    return
                    
            except Exception as e:
                is_last_attempt = attempt == max_retries - 1
                should_retry = should_retry_error(e)
                
                if should_retry and not is_last_attempt:
                    wait_time = calculate_retry_delay(attempt, error=e)
                    logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Request failed permanently: {str(e)}")
                    yield handle_model_client_error(e, params.stream, self._create_error_response)
                    return

    def _convert_stream_chunk(self, chunk: Any) -> StreamingModelResponse:
        """Convert streaming chunk to StreamingModelResponse."""
        delta_content = None
        tool_calls = None
        finish_reason = None
        
        if hasattr(chunk, 'choices') and chunk.choices:
            choice = chunk.choices[0]
            if hasattr(choice, 'delta'):
                delta = choice.delta
                delta_content = getattr(delta, 'content', None)
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    tool_calls = []
                    for tc in delta.tool_calls:
                        tool_calls.append({
                            'id': getattr(tc, 'id', ''),
                            'type': 'function',
                            'function': {
                                'name': getattr(tc.function, 'name', '') if hasattr(tc, 'function') else '',
                                'arguments': getattr(tc.function, 'arguments', '') if hasattr(tc, 'function') else ''
                            }
                        })
            finish_reason = getattr(choice, 'finish_reason', None)
        
        usage = None
        if hasattr(chunk, 'usage') and chunk.usage:
            usage = Usage(
                prompt_tokens=getattr(chunk.usage, 'prompt_tokens', 0),
                completion_tokens=getattr(chunk.usage, 'completion_tokens', 0),
                total_tokens=getattr(chunk.usage, 'total_tokens', 0)
            )
        
        return StreamingModelResponse(
            id=getattr(chunk, 'id', str(uuid.uuid4())),
            created=getattr(chunk, 'created', int(time.time())),
            model=getattr(chunk, 'model', self.cfg.get_aggregated_model_name()),
            choices=[StreamingChoice(
                index=0,
                delta=Message(role='assistant', content=delta_content, tool_calls=tool_calls),
                finish_reason=finish_reason
            )],
            usage=usage
        )

    def _dict_to_model_response(self, completion_dict: Dict[str, Any]) -> ModelResponse:
        """Convert completion dict to ModelResponse."""
        choice_data = completion_dict['choices'][0]
        message_data = choice_data['message']
        
        usage = None
        if 'usage' in completion_dict:
            usage_data = completion_dict['usage']
            usage = Usage(
                prompt_tokens=usage_data.get('prompt_tokens', 0),
                completion_tokens=usage_data.get('completion_tokens', 0),
                total_tokens=usage_data.get('total_tokens', 0)
            )
        
        return ModelResponse(
            id=completion_dict.get('id', str(uuid.uuid4())),
            created=completion_dict.get('created', int(time.time())),
            model=completion_dict.get('model', self.cfg.get_aggregated_model_name()),
            choices=[Choice(
                index=0,
                message=Message(
                    role=message_data['role'],
                    content=message_data.get('content'),
                    tool_calls=message_data.get('tool_calls')
                ),
                finish_reason=choice_data.get('finish_reason', 'stop')
            )],
            usage=usage
        )

    def _fix_tool_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively fix tool schema issues for Azure OpenAI compatibility."""
        if not isinstance(schema, dict):
            return schema
         
        fixed_schema = {}
        for key, value in schema.items():
            if isinstance(value, dict):
                fixed_value = self._fix_tool_schema(value)
                if fixed_value.get("type") == "array" and "items" not in fixed_value:
                    fixed_value["items"] = {"type": "object"}
                fixed_schema[key] = fixed_value
            elif isinstance(value, list):
                fixed_schema[key] = [self._fix_tool_schema(item) if isinstance(item, dict) else item for item in value]
            else:
                fixed_schema[key] = value
        return fixed_schema

    def _build_responses_request_data(self, params: RunParams, responses_items: List[Dict[str, Any]], instructions: str = '') -> Dict[str, Any]:
        """Build request data for litellm.responses() API."""
        model_name = self.cfg.get_actual_model_name()
        
        request_data = {
            'model': model_name,
            'input': responses_items,
            'stream': params.stream,
            'parallel_tool_calls': False  # Disable parallel tool calls
        }
        
        if params.timeout is not None:
            request_data['timeout'] = params.timeout
            request_data['request_timeout'] = params.timeout
        
        request_data['num_retries'] = 0
        
        if instructions:
            request_data['instructions'] = instructions
        
        if params.temperature is not None:
            request_data['temperature'] = params.temperature
        elif self.cfg.temperature is not None:
            request_data['temperature'] = self.cfg.temperature
        
        if params.max_tokens is not None:
            request_data['max_tokens'] = params.max_tokens
        elif self.cfg.max_tokens is not None:
            request_data['max_tokens'] = self.cfg.max_tokens
        
        # Add tools if present - Responses API uses flat format without 'function' wrapper
        if params.tool_params and params.tool_params.tools:
            tools = []
            for tool in params.tool_params.tools:
                if hasattr(tool, 'name'):
                    tools.append({
                        'type': 'function',
                        'name': tool.name,
                        'description': tool.description,
                        'parameters': self._fix_tool_schema(tool.parameters)
                    })
                elif isinstance(tool, dict) and 'function' in tool:
                    fn = tool['function']
                    tools.append({
                        'type': 'function',
                        'name': fn['name'],
                        'description': fn['description'],
                        'parameters': self._fix_tool_schema(fn['parameters'])
                    })
            request_data['tools'] = tools
            
            # Require the model to use a tool when tools are provided
            # Use 'required' instead of {"type": "function"} for Azure Responses API
            request_data['tool_choice'] = 'required'
        
        logger.debug(f"Responses API request: {json.dumps(request_data, indent=2, ensure_ascii=False)}")
        
        # Save request data to trace context for monitoring/debugging
        params.trace_context["llm_request_body"] = request_data
        
        return request_data

    def _create_error_response(self, error_message: str, is_streaming: bool = False) -> Union[ModelResponse, StreamingModelResponse]:
        """Create error response."""
        error_object = {"message": error_message, "type": "api_error", "code": "azure_error"}
        if is_streaming:
            return StreamingModelResponse(error=error_object)
        else:
            return ModelResponse(error=error_object)
