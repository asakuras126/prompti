"""OpenAI Response API client implementation using openai SDK."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Union
from collections.abc import AsyncGenerator, Generator

from openai import AsyncOpenAI, OpenAI
from openai.types import ResponseFormatText, ResponseFormatJSONSchema

from .base import ModelClient, SyncModelClient, ModelConfig, RunParams, should_retry_error, calculate_retry_delay, handle_model_client_error
from .image_utils import convert_image_urls_to_base64
from ..message import Message, ModelResponse, StreamingModelResponse, Usage, Choice, StreamingChoice
from ..logger import get_logger

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
        
        if role == 'system':
            if isinstance(content, str):
                instructions = content
            elif isinstance(content, list):
                text_parts = []
                for seg in content:
                    if isinstance(seg, dict) and seg.get('type') == 'text' and seg.get('text'):
                        text_parts.append(seg['text'])
                if text_parts:
                    instructions = ' '.join(text_parts)
            continue
        
        if role == 'tool':
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
            if content:
                if isinstance(content, str) and content:
                    out.append({'role': 'assistant', 'content': content})
                elif isinstance(content, list):
                    text_parts = []
                    for seg in content:
                        if isinstance(seg, dict) and seg.get('type') == 'text' and seg.get('text'):
                            text_parts.append(seg['text'])
                    if text_parts:
                        out.append({'role': 'assistant', 'content': ' '.join(text_parts)})
            
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
            if isinstance(content, str):
                if content:
                    out.append({'role': role, 'content': content})
            elif isinstance(content, list):
                converted_content = convert_image_urls_to_base64(content, skip_miaoda_files=False)
                message_content = []
                for seg in converted_content:
                    if isinstance(seg, dict):
                        if seg.get('type') == 'text' and seg.get('text'):
                            message_content.append({'type': 'input_text', 'text': seg['text']})
                        elif seg.get('type') == 'image_url' and seg.get('image_url'):
                            image_url_obj = seg['image_url']
                            url = image_url_obj.get('url') if isinstance(image_url_obj, dict) else image_url_obj
                            if url:
                                message_content.append({'type': 'input_image', 'image_url': url})
                
                if message_content:
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
            item_content = getattr(item, 'content', None)
            if item_content and isinstance(item_content, list):
                parts = []
                for seg in item_content:
                    text = getattr(seg, 'text', None)
                    if text:
                        parts.append(text)
                if parts:
                    reasoning_content = '\n\n'.join(parts)
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

    message: Dict[str, Any] = {'role': 'assistant'}
    
    if content:
        message['content'] = content
    else:
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


class OpenAIResponseClient(ModelClient):
    """OpenAI Response API client using openai SDK."""

    provider = "openai_response"

    def __init__(self, cfg: ModelConfig, client=None, is_debug: bool = False) -> None:
        super().__init__(cfg, client, is_debug=is_debug)
        self.api_key = cfg.api_key
        self.api_url = cfg.api_url

        self.openai_client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_url,
            timeout=600.0
        )

    async def _run(self, params: RunParams) -> AsyncGenerator[Union[ModelResponse, StreamingModelResponse], None]:
        """Execute call using OpenAI Response API."""
        max_retries = 2

        for attempt in range(max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1}/{max_retries}")

                messages_dict = [m.to_openai() for m in params.messages]
                responses_items, instructions = messages_to_responses_items(messages_dict)

                request_kwargs = self._build_responses_request_kwargs(params, responses_items, instructions)

                if params.stream:
                    response_stream = await self.openai_client.responses.create(**request_kwargs)
                    async for event in response_stream:
                        chunk = self._convert_stream_chunk(event)
                        if chunk:
                            yield chunk
                    return
                else:
                    responses_result = await self.openai_client.responses.create(**request_kwargs)
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
                    import asyncio
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Request failed permanently: {str(e)}")
                    yield handle_model_client_error(e, params.stream, self._create_error_response)
                    return

    def _convert_stream_chunk(self, event: Any) -> Union[StreamingModelResponse, None]:
        """Convert streaming event to StreamingModelResponse."""
        delta_content = None
        tool_calls = None
        finish_reason = None

        if hasattr(event, 'type'):
            event_type = getattr(event, 'type', None)

            if event_type == 'response.output_text.delta':
                delta_content = getattr(event, 'delta', None)

            elif event_type == 'response.function_call_arguments.delta':
                call_id = getattr(event, 'call_id', '')
                name = getattr(event, 'name', '')
                arguments_delta = getattr(event, 'delta', '')
                if call_id or name or arguments_delta:
                    tool_calls = [{
                        'id': call_id,
                        'type': 'function',
                        'function': {
                            'name': name,
                            'arguments': arguments_delta
                        }
                    }]

            elif event_type == 'response.done':
                finish_reason = 'stop'

        if delta_content is None and tool_calls is None and finish_reason is None:
            return None

        usage = None
        if hasattr(event, 'usage') and event.usage:
            usage = Usage(
                prompt_tokens=getattr(event.usage, 'input_tokens', 0),
                completion_tokens=getattr(event.usage, 'output_tokens', 0),
                total_tokens=getattr(event.usage, 'total_tokens', 0)
            )

        event_id = getattr(event, 'response_id', None) or str(uuid.uuid4())

        return StreamingModelResponse(
            id=event_id,
            created=int(time.time()),
            model=self.cfg.get_aggregated_model_name(),
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
        """Recursively fix tool schema issues for OpenAI compatibility."""
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

    def _build_responses_request_kwargs(self, params: RunParams, responses_items: List[Dict[str, Any]], instructions: str = '') -> Dict[str, Any]:
        """Build request kwargs for openai.responses.create() API."""
        model_name = self.cfg.get_actual_model_name()

        request_kwargs = {
            'model': model_name,
            'input': responses_items,
            'stream': params.stream,
        }

        if params.timeout is not None:
            request_kwargs['timeout'] = params.timeout

        if instructions:
            request_kwargs['instructions'] = instructions

        if params.temperature is not None:
            request_kwargs['temperature'] = params.temperature
        elif self.cfg.temperature is not None:
            request_kwargs['temperature'] = self.cfg.temperature

        # Note: Azure OpenAI Responses API may not support max_tokens parameter
        # if params.max_tokens is not None:
        #     request_kwargs['max_tokens'] = params.max_tokens
        # elif self.cfg.max_tokens is not None:
        #     request_kwargs['max_tokens'] = self.cfg.max_tokens

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
            request_kwargs['tools'] = tools
            request_kwargs['tool_choice'] = 'required'

        # Support reasoning.effort parameter via ModelConfig.ext
        if self.cfg.ext and 'reasoning_effort' in self.cfg.ext:
            reasoning_effort = self.cfg.ext['reasoning_effort']
            if reasoning_effort in ('low', 'medium', 'high'):
                request_kwargs['reasoning'] = {'effort': reasoning_effort}
            else:
                logger.warning(f"Invalid reasoning_effort value: {reasoning_effort}. Must be one of: low, medium, high")

        logger.debug(f"Responses API request: {request_kwargs}")

        params.trace_context["llm_request_body"] = request_kwargs

        return request_kwargs

    def _create_error_response(self, error_message: str, is_streaming: bool = False, error_type: str = None, error_code: str = None) -> Union[ModelResponse, StreamingModelResponse]:
        """Create error response."""
        error_object = {
            "message": error_message, 
            "type": error_type or "api_error", 
            "code": error_code or "openai_response_error"
        }
        if is_streaming:
            return StreamingModelResponse(error=error_object)
        else:
            return ModelResponse(error=error_object)


class SyncOpenAIResponseClient(SyncModelClient):
    """Synchronous OpenAI Response API client using openai SDK."""

    provider = "openai_response"

    def __init__(self, cfg: ModelConfig, client=None, is_debug: bool = False) -> None:
        super().__init__(cfg, client, is_debug=is_debug)
        self.api_key = cfg.api_key
        self.api_url = cfg.api_url
        logger.debug(f"api_url: {self.api_url}")
        self.openai_client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_url,
            timeout=600.0
        )

    def _run(self, params: RunParams) -> Generator[Union[ModelResponse, StreamingModelResponse], None, None]:
        """Execute call using OpenAI Response API."""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1}/{max_retries}")
                
                messages_dict = [m.to_openai() for m in params.messages]
                responses_items, instructions = messages_to_responses_items(messages_dict)
                
                request_kwargs = self._build_responses_request_kwargs(params, responses_items, instructions)
                
                if params.stream:
                    response_stream = self.openai_client.responses.create(**request_kwargs)
                    for event in response_stream:
                        chunk = self._convert_stream_chunk(event)
                        if chunk:
                            yield chunk
                    return
                else:
                    logger.debug(f"request kwargs: {request_kwargs}")
                    responses_result = self.openai_client.responses.create(**request_kwargs)
                    logger.debug(f"responses_result type: {type(responses_result)}")
                    logger.debug(f"responses_result has output: {hasattr(responses_result, 'output')}")
                    logger.debug(f"Raw responses output result: {responses_result.output if hasattr(responses_result, 'output') else None}")
                    
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

    def _convert_stream_chunk(self, event: Any) -> Union[StreamingModelResponse, None]:
        """Convert streaming event to StreamingModelResponse."""
        delta_content = None
        tool_calls = None
        finish_reason = None
        
        if hasattr(event, 'type'):
            event_type = getattr(event, 'type', None)
            
            if event_type == 'response.output_text.delta':
                delta_content = getattr(event, 'delta', None)
            
            elif event_type == 'response.function_call_arguments.delta':
                call_id = getattr(event, 'call_id', '')
                name = getattr(event, 'name', '')
                arguments_delta = getattr(event, 'delta', '')
                if call_id or name or arguments_delta:
                    tool_calls = [{
                        'id': call_id,
                        'type': 'function',
                        'function': {
                            'name': name,
                            'arguments': arguments_delta
                        }
                    }]
            
            elif event_type == 'response.done':
                finish_reason = 'stop'
        
        if delta_content is None and tool_calls is None and finish_reason is None:
            return None
        
        usage = None
        if hasattr(event, 'usage') and event.usage:
            usage = Usage(
                prompt_tokens=getattr(event.usage, 'input_tokens', 0),
                completion_tokens=getattr(event.usage, 'output_tokens', 0),
                total_tokens=getattr(event.usage, 'total_tokens', 0)
            )
        
        event_id = getattr(event, 'response_id', None) or str(uuid.uuid4())
        
        return StreamingModelResponse(
            id=event_id,
            created=int(time.time()),
            model=self.cfg.get_aggregated_model_name(),
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
        """Recursively fix tool schema issues for OpenAI compatibility."""
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

    def _build_responses_request_kwargs(self, params: RunParams, responses_items: List[Dict[str, Any]], instructions: str = '') -> Dict[str, Any]:
        """Build request kwargs for openai.responses.create() API."""
        model_name = self.cfg.get_actual_model_name()

        request_kwargs = {
            'model': model_name,
            'input': responses_items,
            'stream': params.stream,
        }

        if params.timeout is not None:
            request_kwargs['timeout'] = params.timeout

        if instructions:
            request_kwargs['instructions'] = instructions

        if params.temperature is not None:
            request_kwargs['temperature'] = params.temperature
        elif self.cfg.temperature is not None:
            request_kwargs['temperature'] = self.cfg.temperature

        # Note: Azure OpenAI Responses API may not support max_tokens parameter
        # if params.max_tokens is not None:
        #     request_kwargs['max_tokens'] = params.max_tokens
        # elif self.cfg.max_tokens is not None:
        #     request_kwargs['max_tokens'] = self.cfg.max_tokens

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
            request_kwargs['tools'] = tools
            request_kwargs['tool_choice'] = 'required'

        # Support reasoning.effort parameter via ModelConfig.ext
        if self.cfg.ext and 'reasoning_effort' in self.cfg.ext:
            reasoning_effort = self.cfg.ext['reasoning_effort']
            if reasoning_effort in ('low', 'medium', 'high'):
                request_kwargs['reasoning'] = {'effort': reasoning_effort}
            else:
                logger.warning(f"Invalid reasoning_effort value: {reasoning_effort}. Must be one of: low, medium, high")

        logger.debug(f"Responses API request: {request_kwargs}")

        params.trace_context["llm_request_body"] = request_kwargs

        return request_kwargs

    def _create_error_response(self, error_message: str, is_streaming: bool = False, error_type: str = None, error_code: str = None) -> Union[ModelResponse, StreamingModelResponse]:
        """Create error response."""
        error_object = {
            "message": error_message, 
            "type": error_type or "api_error", 
            "code": error_code or "openai_response_error"
        }
        if is_streaming:
            return StreamingModelResponse(error=error_object)
        else:
            return ModelResponse(error=error_object)
