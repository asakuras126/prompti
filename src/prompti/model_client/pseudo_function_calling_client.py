"""Pseudo Function Calling Client implementation.

This client wraps any OpenAI-compatible API and adds pseudo function calling support.
When tool calls are requested, it converts them to non-tool call format internally,
makes the request to the underlying model, then converts the response back to tool call format.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
import time
import shlex
from collections.abc import AsyncGenerator, Generator
from typing import Any, Dict, Union, Optional
from datetime import datetime, timezone

import httpx

from ..message import Message, ModelResponse, StreamingModelResponse, Choice, StreamingChoice, Usage
from .base import (
    ModelClient, SyncModelClient, RunParams, ModelConfig,
    should_retry_error, calculate_retry_delay, handle_model_client_error, is_context_length_error
)
from .image_utils import convert_image_urls_to_base64
from ..logger import get_logger

logger = get_logger(__name__)

# Tool call conversion functions adapted from fn_call_converter.py
SYSTEM_PROMPT_SUFFIX_TEMPLATE = """
You have access to the following functions:

{description}

If you choose to call a function ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after.
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>
"""

TOOL_CALL_EXAMPLE_PREFIX = """
Here's a running example of how to perform a task with the provided tools.
 
--------------------- START OF EXAMPLE ---------------------
 
USER: Create a list of numbers from 1 to 10, and display them in a web page at port 5000.
 

ASSISTANT: Sure! Let me first check the current directory:
<function=execute_bash>
<parameter=command>
pwd && ls
</parameter>
</function>
 
USER: EXECUTION RESULT of [execute_bash]:
/workspace
openhands@runtime:~/workspace$

ASSISTANT:
There is no `app.py` file in the current directory. Let me create a Python file `app.py`:
<function=str_replace_editor>
<parameter=command>create</parameter>
<parameter=path>/workspace/app.py</parameter>
<parameter=file_text>
from flask import Flask
app = Flask(__name__)
 
@app.route('/')
def index():
    numbers = list(range(1, 11))
    return str(numbers)
 
if __name__ == '__main__':
    app.run(port=5000)
</parameter>
</function>
 
USER: EXECUTION RESULT of [str_replace_editor]:
File created successfully at: /workspace/app.py

ASSISTANT:
Let me run the Python file for you:
<function=execute_bash>
<parameter=command>
python3 app.py > server.log 2>&1 &
</parameter>
</function>
 
USER: EXECUTION RESULT of [execute_bash]:
[1] 121
[1]+  Exit 1                  python3 app.py > server.log 2>&1
 
ASSISTANT:
Looks like the server was running with PID 121 then crashed. Let me check the server log:
<function=execute_bash>
<parameter=command>
cat server.log
</parameter>
</function>
 
USER: EXECUTION RESULT of [execute_bash]:
Traceback (most recent call last):
  File "/workspace/app.py", line 2, in <module>
    from flask import Flask
ModuleNotFoundError: No module named 'flask'
 
ASSISTANT:
Looks like the server crashed because the `flask` module is not installed. Let me install the `flask` module for you:
<function=execute_bash>
<parameter=command>
pip3 install flask
</parameter>
</function>
 
USER: EXECUTION RESULT of [execute_bash]:
Defaulting to user installation because normal site-packages is not writeable
Collecting flask
  Using cached flask-3.0.3-py3-none-any.whl (101 kB)
Collecting blinker>=1.6.2
  Using cached blinker-1.7.0-py3-none-any.whl (13 kB)
Collecting Werkzeug>=3.0.0
  Using cached werkzeug-3.0.2-py3-none-any.whl (226 kB)
Collecting click>=8.1.3
  Using cached click-8.1.7-py3-none-any.whl (97 kB)
Collecting itsdangerous>=2.1.2
  Using cached itsdangerous-2.2.0-py3-none-any.whl (16 kB)
Requirement already satisfied: Jinja2>=3.1.2 in /home/openhands/.local/lib/python3.10/site-packages (from flask) (3.1.3)
Requirement already satisfied: MarkupSafe>=2.0 in /home/openhands/.local/lib/python3.10/site-packages (from Jinja2>=3.1.2->flask) (2.1.5)
Installing collected packages: Werkzeug, itsdangerous, click, blinker, flask
Successfully installed Werkzeug-3.0.2 blinker-1.7.0 click-8.1.7 flask-3.0.3 itsdangerous-2.2.0
 
ASSISTANT:
The `flask` module is installed successfully. Let me run the Python file again:
<function=execute_bash>
<parameter=command>
python3 app.py > server.log 2>&1 &
</parameter>
</function>
 
USER: EXECUTION RESULT of [execute_bash]:
[1] 124
 
ASSISTANT:
Let me check the server log again:
<function=execute_bash>
<parameter=command>
cat server.log
</parameter>
</function>
 
USER: EXECUTION RESULT of [execute_bash]:
* Serving Flask app 'app'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 
ASSISTANT:
The server is running on port 5000 with PID 124. You can access the list of numbers by visiting http://127.0.0.1:5000. If you have any further questions, feel free to ask!

USER: Now kill the server, make it display the numbers in a table format.
 
ASSISTANT:
Sure! Let me stop the server first:
<function=execute_bash>
<parameter=command>
kill 124
</parameter>
</function>
 
USER: EXECUTION RESULT of [execute_bash]:
[1]+  Terminated              python3 app.py > server.log 2>&1

ASSISTANT:
Now let me display the numbers in a table format:
<function=str_replace_editor>
<parameter=command>str_replace</parameter>
<parameter=path>/workspace/app.py</parameter>
<parameter=old_str>return str(numbers)</parameter>
<parameter=new_str>return '<table>' + ''.join([f'<tr><td>{i}</td></tr>' for i in numbers]) + '</table>'</parameter>
</function>
 
USER: EXECUTION RESULT of [str_replace_editor]:
The file /workspace/app.py has been edited. Here's the result of running `cat -n` on a snippet of /workspace/app.py:
     3
     4  @app.route('/')
     5  def index():
     6      numbers = list(range(1, 11))
     7      return '<table>' + ''.join([f'<tr><td>{i}</td></tr>' for i in numbers]) + '</table>'
     8
     9  if __name__ == '__main__':
    10      app.run(port=5000)
Review the changes and make sure they are as expected. Edit the file again if necessary.
 

ASSISTANT:
Running the updated file:
<function=execute_bash>
<parameter=command>
python3 app.py > server.log 2>&1 &
</parameter>
</function>
 
USER: EXECUTION RESULT of [execute_bash]:
[1] 126
 
ASSISTANT:
The server is running on port 5000 with PID 126. You can access the list of numbers in a table format by visiting http://127.0.0.1:5000.

ASSISTANT:
The server is running on port 5000 with PID 126. You can access the list of numbers in a table format by visiting http://127.0.0.1:5000. Let me know if you have any further requests!
<function=finish>
<parameter=command>submit</parameter>
<parameter=result>
# First section: Summary for User
    A user-facing explanation. Since the user has no knowledge of code, programming, architecture, or the runtime environment,
    the description should be simple and easy to understand, avoiding any technical jargon to facilitate quick comprehension.
# Second section: List of changes
    This is the Pull Request (PR) description, which should detail all changes made so far.
</parameter>
</function>

--------------------- END OF EXAMPLE ---------------------
 
Do NOT assume the environment is the same as in the example above.
 
--------------------- NEW TASK DESCRIPTION ---------------------
"""

TOOL_CALL_EXAMPLE_SUFFIX="""
--------------------- END OF NEW TASK DESCRIPTION ---------------------
 
PLEASE follow the format strictly! PLEASE EMIT ONE AND ONLY ONE FUNCTION CALL PER MESSAGE.
"""
STOP_WORDS = ['</function']

# Regex patterns for function call parsing
FN_REGEX_PATTERN = r'<function=([^>]+)>(.*?)</function>'
FN_PARAM_REGEX_PATTERN = r'<parameter=([^>]+)>(.*?)</parameter>'


def _merge_consecutive_user_messages(messages: list[dict]) -> list[dict]:
    """Merge consecutive user messages using '\\n\\n' separator."""
    if not messages:
        return messages
    
    merged_messages = []
    current_user_content = []
    
    for message in messages:
        role = message['role']
        content = message['content']
        
        if role == 'user':
            # Collect consecutive user messages
            if isinstance(content, str):
                current_user_content.append(content)
            elif isinstance(content, list):
                # Handle multimodal content - extract text parts
                text_parts = []
                non_text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item['text'])
                    else:
                        non_text_parts.append(item)
                
                if text_parts:
                    current_user_content.append('\n\n'.join(text_parts))
                
                # If there are non-text parts (like images), we need to flush current content
                if non_text_parts:
                    if current_user_content:
                        # Create merged text message
                        merged_text = '\n\n'.join(current_user_content)
                        merged_content = [{'type': 'text', 'text': merged_text}] + non_text_parts
                        merged_messages.append({'role': 'user', 'content': merged_content})
                        current_user_content = []
                    else:
                        # Just add the current message as is
                        merged_messages.append(message)
        else:
            # Non-user message, flush any accumulated user content
            if current_user_content:
                merged_text = '\n\n'.join(current_user_content)
                merged_messages.append({'role': 'user', 'content': merged_text})
                current_user_content = []
            
            merged_messages.append(message)
    
    # Handle any remaining user content at the end
    if current_user_content:
        merged_text = '\n\n'.join(current_user_content)
        merged_messages.append({'role': 'user', 'content': merged_text})
    
    return merged_messages


def convert_tools_to_description(tools: list[dict]) -> str:
    """Convert tools to text description for prompt injection."""
    ret = ''
    for i, tool in enumerate(tools):
        # Support both formats: {"type": "function", "function": {...}} and direct tool definition
        if 'type' in tool and tool['type'] == 'function':
            fn = tool['function']
        else:
            # Direct tool definition format
            fn = tool
            
        if i > 0:
            ret += '\n'
        ret += f'---- BEGIN FUNCTION #{i + 1}: {fn["name"]} ----\n'
        ret += f'Description: {fn["description"]}\n'

        if 'parameters' in fn:
            ret += 'Parameters:\n'
            properties = fn['parameters'].get('properties', {})
            required_params = set(fn['parameters'].get('required', []))

            for j, (param_name, param_info) in enumerate(properties.items()):
                is_required = param_name in required_params
                param_status = 'required' if is_required else 'optional'
                param_type = param_info.get('type', 'string')
                desc = param_info.get('description', 'No description provided')

                if 'enum' in param_info:
                    enum_values = ', '.join(f'`{v}`' for v in param_info['enum'])
                    desc += f'\nAllowed values: [{enum_values}]'

                ret += f'  ({j + 1}) {param_name} ({param_type}, {param_status}): {desc}\n'
        else:
            ret += 'No parameters are required for this function.\n'

        ret += f'---- END FUNCTION #{i + 1} ----\n'
    return ret


def convert_fncall_messages_to_non_fncall_messages(messages: list[dict], tools: list[dict], task_description: str = "") -> list[dict]:
    """Convert function calling messages to non-function calling messages."""
    messages = copy.deepcopy(messages)
    
    if not tools:
        return messages

    formatted_tools = convert_tools_to_description(tools)
    system_prompt_suffix = SYSTEM_PROMPT_SUFFIX_TEMPLATE.format(description=formatted_tools)

    converted_messages = []
    system_message_added = False
    last_assistant_function_name = None  # Track the last function name from assistant message
    for message in messages:
        role = message['role']
        content = message['content']

        # System messages: append system prompt suffix
        if role == 'system':
            if isinstance(content, str):
                content += system_prompt_suffix
            elif isinstance(content, list):
                if content and content[-1]['type'] == 'text':
                    content[-1]['text'] += system_prompt_suffix
                else:
                    content.append({'type': 'text', 'text': system_prompt_suffix})
            converted_messages.append({'role': 'system', 'content': content})
            system_message_added = True

        # User messages: add TOOL_CALL_EXAMPLE after system but before first user
        elif role == 'user':
            # Add TOOL_CALL_EXAMPLE before the first user message if system was added
            if system_message_added and len(converted_messages) > 0 and converted_messages[-1]['role'] == 'system':
                # Format TOOL_CALL_EXAMPLE with task_description
                if isinstance(content, str):
                    content_str = content
                elif isinstance(content, list) and content and content[-1]['type'] == 'text':
                    content_str = content[-1]['text']
                else:
                    content_str = ""
                # formatted_example = TOOL_CALL_EXAMPLE_PREFIX + content_str + TOOL_CALL_EXAMPLE_SUFFIX
                # 模型研发部同学告知TOOL_CALL_EXAMPLE_PREFIX需要移除
                formatted_example = content_str
                converted_messages.append({'role': 'user', 'content': formatted_example})
                system_message_added = False  # Only add once
            else:
                converted_messages.append({'role': 'user', 'content': content})

        # Assistant messages: convert tool calls to text format
        elif role == 'assistant':
            if 'tool_calls' in message and message['tool_calls'] is not None:
                if len(message['tool_calls']) != 1:
                    raise ValueError(f'Expected exactly one tool call, got {len(message["tool_calls"])}')
                
                tool_call = message['tool_calls'][0]
                tool_content = convert_tool_call_to_string(tool_call)
                
                if isinstance(content, str):
                    content += '\n\n' + tool_content if content else tool_content
                elif isinstance(content, list):
                    if content and content[-1]['type'] == 'text':
                        content[-1]['text'] += '\n\n' + tool_content
                    else:
                        content.append({'type': 'text', 'text': tool_content})

            # Extract function name from assistant message content for pseudo function calling format
            function_name = _extract_function_name_from_content(content)
            if function_name:
                last_assistant_function_name = function_name
            converted_messages.append({'role': 'assistant', 'content': content})

        # Tool messages: convert to user messages with execution result format
        elif role == 'tool':
            tool_name = last_assistant_function_name or 'function'
            prefix = f'EXECUTION RESULT of [{tool_name}]:\n'
            last_assistant_function_name = None
            
            if isinstance(content, str):
                content = prefix + content
            elif isinstance(content, list):
                if content and content[0].get('type') == 'text':
                    content[0]['text'] = prefix + content[0]['text']
                else:
                    content = [{'type': 'text', 'text': prefix}] + content
            
            converted_messages.append({'role': 'user', 'content': content})

    # After processing all messages and adding TOOL_CALL_EXAMPLE, merge consecutive user messages
    converted_messages = _merge_consecutive_user_messages(converted_messages)

    return converted_messages


def _extract_function_name_from_content(content) -> str | None:
    """Extract function name from assistant message content that contains <function=name> tags."""
    text_content = ""
    
    if isinstance(content, str):
        text_content = content
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                text_content += item.get('text', '')
    
    # Use regex to find <function=name> pattern
    match = re.search(r'<function=([^>]+)>', text_content)
    if match:
        return match.group(1)
    
    return None


def _fix_incomplete_functions(content) -> tuple[any, bool]:
    """Fix incomplete function calls by adding missing </function> tags.
    
    Returns:
        tuple: (fixed_content, was_modified)
    """
    if isinstance(content, str):
        text_content = content
        was_list = False
    elif isinstance(content, list):
        text_content = ""
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                text_content += item.get('text', '')
        was_list = True
    else:
        return content, False
    
    original_text = text_content
    
    # Find all function start tags
    function_starts = list(re.finditer(r'<function=([^>]+)>', text_content))
    if not function_starts:
        return content, False
    
    # Process functions from end to beginning to avoid position shifts
    for i in reversed(range(len(function_starts))):
        start_match = function_starts[i]
        func_name = start_match.group(1)
        start_pos = start_match.start()
        
        # Find the next function start or end of text to determine search boundary
        next_start_pos = len(text_content)
        if i + 1 < len(function_starts):
            next_start_pos = function_starts[i + 1].start()
        
        # Look for closing tag within this function's scope
        search_text = text_content[start_pos:next_start_pos]
        if '</function>' not in search_text:
            # This function is incomplete, add closing tag
            # Find the end of this function's content (before next function or end of text)
            insert_pos = next_start_pos
            # Insert the missing closing tag
            text_content = text_content[:insert_pos] + '\n</function>' + text_content[insert_pos:]
    
    # Update the content with fixed text
    if text_content != original_text:
        if was_list:
            # Update the list content
            if isinstance(content, list):
                new_content = []
                text_updated = False
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text' and not text_updated:
                        new_item = item.copy()
                        new_item['text'] = text_content
                        new_content.append(new_item)
                        text_updated = True
                    else:
                        new_content.append(item)
                return new_content, True
        else:
            # Return fixed string
            return text_content, True
    
    return content, False


def convert_tool_call_to_string(tool_call: dict) -> str:
    """Convert tool call to string format."""
    ret = f'<function={tool_call["function"]["name"]}>\n'
    try:
        args = json.loads(tool_call['function']['arguments'])
    except json.JSONDecodeError as e:
        raise ValueError(f'Failed to parse tool call arguments: {e}')
    
    for param_name, param_value in args.items():
        ret += f'<parameter={param_name}>'
        if isinstance(param_value, (list, dict)):
            ret += json.dumps(param_value)
        else:
            ret += str(param_value)
        ret += '</parameter>\n'
    ret += '</function>'
    return ret


def _fix_stopword(content: str) -> str:
    """Fix incomplete function call stopwords."""
    if '<function=' in content and content.count('<function=') == 1:
        if content.endswith('</'):
            content = content.rstrip() + 'function>'
        elif not content.endswith('</function>'):
            content = content + '\n</function>'
    return content


def convert_non_fncall_messages_to_fncall_messages(messages: list[dict], tools: list[dict]) -> tuple[list[dict], str | None]:
    """Convert non-function calling messages back to function calling messages.
    
    Returns:
        tuple: (converted_messages, error_message)
        - converted_messages: List of converted messages
        - error_message: Error message if parsing failed, None if successful
    """
    messages = copy.deepcopy(messages)
    converted_messages = []
    tool_call_counter = 1

    for message in messages:
        role, content = message['role'], message.get('content', '')

        if role == 'assistant':
            if isinstance(content, str):
                # Check if content contains function tags
                has_function_tag = '<function=' in content
                
                if has_function_tag:
                    # Try to fix incomplete functions first
                    content, was_modified = _fix_incomplete_functions(content)
                    
                    # Then try to fix stopwords
                    content = _fix_stopword(content)
                
                # Find all function matches (support multiple functions)
                fn_matches = list(re.finditer(FN_REGEX_PATTERN, content, re.DOTALL))
                
                if has_function_tag and not fn_matches:
                    # Has function tags but still no valid matches after fixing - parsing failed
                    return converted_messages, "Failed to parse function call: invalid function format detected"
                
                if fn_matches:
                    # Try to parse function matches until we find one that works
                    successful_parse = False
                    last_error = None
                    
                    for fn_match in fn_matches:
                        fn_name = fn_match.group(1)
                        fn_body = fn_match.group(2)
                        
                        try:
                            # Find matching tool - support both formats
                            matching_tool = None
                            for tool in tools:
                                if 'type' in tool and tool['type'] == 'function':
                                    if tool['function']['name'] == fn_name:
                                        matching_tool = tool['function']
                                        break
                                else:
                                    # Direct tool definition format
                                    if tool['name'] == fn_name:
                                        matching_tool = tool
                                        break
                            
                            # Skip if tool not found, but continue trying other functions
                            # if not matching_tool:
                            #     last_error = f"Function '{fn_name}' not found in available tools"
                            #     continue

                            # Parse parameters
                            param_matches = re.finditer(FN_PARAM_REGEX_PATTERN, fn_body, re.DOTALL)
                            params = {}
                            for param_match in param_matches:
                                param_name = param_match.group(1)
                                param_value = param_match.group(2).strip()
                                
                                # Try to parse as JSON for complex types
                                try:
                                    param_value = json.loads(param_value)
                                except json.JSONDecodeError:
                                    # Keep as string if not JSON
                                    pass
                                
                                params[param_name] = param_value

                            # Check if this function has valid parameters
                            # Skip functions with no parameters (likely incomplete)
                            # if not params:
                            #     last_error = f"Function '{fn_name}' has no valid parameters, skipping"
                            #     continue

                            # Create tool call
                            tool_call_id = f'toolu_{tool_call_counter:02d}'
                            tool_call = {
                                'id': tool_call_id,
                                'type': 'function',
                                'function': {'name': fn_name, 'arguments': json.dumps(params, ensure_ascii=False)},
                            }
                            tool_call_counter += 1

                            # Remove function call from content (only this successful one)
                            content = content[:fn_match.start()].strip()
                            
                            converted_messages.append({
                                'role': 'assistant', 
                                'content': content, 
                                'tool_calls': [tool_call]
                            })
                            
                            successful_parse = True
                            break  # Successfully parsed, stop trying other functions
                            
                        except Exception as e:
                            last_error = f"Failed to parse function '{fn_name}': {str(e)}"
                            continue  # Try next function
                    
                    # If no function was successfully parsed, return error
                    if not successful_parse:
                        error_msg = last_error or "Failed to parse any function call"
                        return converted_messages, error_msg
                else:
                    converted_messages.append(message)
            else:
                fn_match = None
                converted_messages.append(message)
        else:
            converted_messages.append(message)

    return converted_messages, None


class PseudoFunctionCallingClient(ModelClient):
    """Client that adds pseudo function calling support to any OpenAI-compatible API."""

    provider = "pseudo_function_calling"

    def __init__(self, cfg: ModelConfig, client: httpx.AsyncClient | None = None, is_debug: bool = False) -> None:
        """Initialize the client with configuration."""
        super().__init__(cfg, client, is_debug=is_debug)

    def _generate_curl_command(self, url: str, headers: Dict[str, str], request_data: Dict[str, Any]) -> str:
        """Generate equivalent curl command for debugging."""
        curl_parts = ["curl", "-X", "POST"]
        
        # Add headers
        for key, value in headers.items():
            curl_parts.extend(["-H", f"{key}: {value}"])
        
        # Add JSON data
        json_data = json.dumps(request_data, ensure_ascii=False)
        curl_parts.extend(["-d", json_data])
        
        # Add URL
        curl_parts.append(url)
        
        # Use shlex.join for proper escaping (Python 3.8+)
        try:
            return shlex.join(curl_parts)
        except AttributeError:
            # Fallback for older Python versions
            return " ".join(shlex.quote(part) for part in curl_parts)

    async def _run(self, params: RunParams) -> AsyncGenerator[Union[ModelResponse, StreamingModelResponse], None]:
        """Execute the API call with pseudo function calling support."""
        # Initialize basic trace context - will be updated with actual call details later
        params.trace_context["responses"] = []
        
        # Check if we have tool calls to handle
        has_tools = params.tool_params and getattr(params.tool_params, 'tools', None)
        
        if not has_tools:
            # No tools, but still merge consecutive user messages
            messages = [msg.to_openai() for msg in params.messages]
            merged_messages = _merge_consecutive_user_messages(messages)
            
            # Create modified params with merged messages
            no_tools_params = copy.deepcopy(params)
            no_tools_params.messages = [Message.from_openai(msg) for msg in merged_messages]
            
            async for response in self._make_regular_call(no_tools_params):
                yield response
                return

        # Convert messages to pseudo function calling format
        tools = [tool.model_dump() if hasattr(tool, 'model_dump') else tool 
                for tool in params.tool_params.tools]
        
        # Convert messages format
        messages = [msg.to_openai() for msg in params.messages]
        converted_messages = convert_fncall_messages_to_non_fncall_messages(messages, tools)
        
        # Create modified params for pseudo FC call
        pseudo_params = copy.deepcopy(params)
        pseudo_params.messages = [Message.from_openai(msg) for msg in converted_messages]
        pseudo_params.tool_params = None  # Remove tools since they're now in the prompt
        
        # Add stop words
        if pseudo_params.stop:
            if isinstance(pseudo_params.stop, str):
                pseudo_params.stop = [pseudo_params.stop] + STOP_WORDS
            else:
                pseudo_params.stop = list(pseudo_params.stop) + STOP_WORDS
        else:
            pseudo_params.stop = STOP_WORDS

        # Make the call and try to parse response
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                async for response in self._make_regular_call(pseudo_params):
                    # Check if this is already an error response
                    if (hasattr(response, 'choices') and response.choices and 
                        response.choices[0].finish_reason == "error"):
                        # This is an error response, don't retry, just return it
                        yield response
                        return
                    
                    # Try to convert response back to function calling format
                    converted_response = await self._convert_response_to_function_calling(
                        response, tools, retry_count < max_retries, params.trace_context
                    )
                    
                    if converted_response:
                        yield converted_response
                        return
                    else:
                        # Failed to parse, retry if possible
                        if retry_count < max_retries:
                            retry_count += 1
                            logger.warning(f"Failed to parse function call, retrying ({retry_count}/{max_retries})")
                            continue
                        else:
                            # Return original response if parsing fails
                            logger.warning("Failed to parse function call, returning original response")
                            yield response
                            return
                            
            except Exception as e:
                if retry_count < max_retries:
                    retry_count += 1
                    wait_time = calculate_retry_delay(retry_count - 1, error=e)
                    logger.warning(f"Request failed, retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Final error handling
                    logger.error(f"All retries failed: {str(e)}")
                    yield handle_model_client_error(e, params.stream, self._create_error_response)
                    return

    async def _make_regular_call(self, params: RunParams) -> AsyncGenerator[Union[ModelResponse, StreamingModelResponse], None]:
        """Make a regular OpenAI-compatible API call."""
        request_data = self._build_request_data(params)
        url = self.cfg.api_url or "https://api.openai.com/v1/chat/completions"
        headers = self._build_headers()
        
        # Generate and log curl command
        curl_command = self._generate_curl_command(url, headers, request_data)
        logger.debug(f"Equivalent curl command:\n{curl_command}")
        
        timeout = params.timeout if params.timeout is not None else 600

        try:
            if params.stream:
                async with self._client.stream("POST", url, headers=headers, json=request_data, timeout=timeout) as response:
                    response.raise_for_status()
                    async for message in self._process_streaming_response(response):
                        yield message
            else:
                response = await self._client.post(url=url, headers=headers, json=request_data, timeout=timeout)
                response.raise_for_status()
                yield self._process_non_streaming_response(response, params.trace_context)
        except Exception as e:
            # Handle HTTP errors and timeouts
            logger.error(f"HTTP request failed: {str(e)}")
            yield handle_model_client_error(e, params.stream, self._create_error_response)
            return


    async def _convert_response_to_function_calling(self, response: Union[ModelResponse, StreamingModelResponse], 
                                                   tools: list[dict], allow_retry: bool = True, trace_context: dict = None) -> Optional[Union[ModelResponse, StreamingModelResponse]]:
        """Convert pseudo function calling response back to standard format."""
        if isinstance(response, StreamingModelResponse):
            # For streaming responses, we need to collect the full content first
            # This is a limitation of the pseudo FC approach
            logger.warning("Streaming not fully supported with pseudo function calling conversion")
            return response
            
        if not (hasattr(response, 'choices') and response.choices and response.choices[0].message):
            return response

        try:
            logger.info(response)
            content = response.choices[0].message.content or ""
            
            # Record original pseudo function calling response before conversion
            if trace_context is not None:
                original_pseudo_fc_data = {
                    "id": response.id,
                    "object": response.object,
                    "created": response.created,
                    "model": response.model,
                    "choices": [{
                        "index": choice.index,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content,
                            "tool_calls": None  # No tool calls in pseudo FC response
                        },
                        "finish_reason": choice.finish_reason
                    } for choice in response.choices],
                    "usage": response.usage.model_dump() if hasattr(response.usage, 'model_dump') else response.usage.__dict__ if response.usage else None
                }
                trace_context["llm_response_body_pseudo_fc"] = original_pseudo_fc_data
            
            # Try to parse function call
            temp_messages = [{"role": "assistant", "content": content}]
            converted_back, parse_error = convert_non_fncall_messages_to_fncall_messages(temp_messages, tools)
            
            if parse_error:
                # Parsing failed - return error response instead of retrying
                logger.error(f"Tool call parsing failed: {parse_error}")
                error_response = self._create_error_response(parse_error, False)
                return error_response
            
            if converted_back and converted_back[0].get("tool_calls"):
                # Successfully parsed tool calls
                tool_calls = converted_back[0]["tool_calls"]
                response.choices[0].message.tool_calls = tool_calls
                response.choices[0].message.content = converted_back[0]["content"]
                
                # Record converted response in trace context
                if trace_context is not None:
                    # Create a dict representation of the converted response
                    converted_data = {
                        "id": response.id,
                        "object": response.object,
                        "created": response.created,
                        "model": response.model,
                        "choices": [{
                            "index": choice.index,
                            "message": {
                                "role": choice.message.role,
                                "content": choice.message.content,
                                "tool_calls": [tc.model_dump() if hasattr(tc, 'model_dump') else tc for tc in choice.message.tool_calls] if choice.message.tool_calls else None
                            },
                            "finish_reason": choice.finish_reason
                        } for choice in response.choices],
                        "usage": response.usage.model_dump() if hasattr(response.usage, 'model_dump') else response.usage.__dict__ if response.usage else None
                    }
                    trace_context["llm_response_body_converted"] = converted_data
                
                return response
            else:
                # No tool calls found
                return None if allow_retry else response
                
        except Exception as e:
            logger.error(f"Failed to convert pseudo FC response: {str(e)}")
            # Return error response instead of original response to ensure proper error handling
            error_response = self._create_error_response(f"Response parsing failed: {str(e)}", False)
            return error_response

    def _build_request_data(self, params: RunParams) -> Dict[str, Any]:
        """Build request data for OpenAI-compatible API."""
        messages = []
        for m in params.messages:
            openai_msg = m.to_openai()
            if 'content' in openai_msg and isinstance(openai_msg['content'], list):
                openai_msg['content'] = convert_image_urls_to_base64(openai_msg['content'])
            messages.append(openai_msg)

        request_data = {
            "model": self.cfg.get_actual_model_name(),
            "messages": messages,
            "stream": params.stream,
        }

        # Add parameters if set
        if params.temperature is not None:
            request_data["temperature"] = params.temperature
        elif self.cfg.temperature is not None:
            request_data["temperature"] = self.cfg.temperature

        if params.max_tokens is not None:
            request_data["max_tokens"] = params.max_tokens
        elif self.cfg.max_tokens is not None:
            request_data["max_tokens"] = self.cfg.max_tokens

        if params.top_p is not None:
            request_data["top_p"] = params.top_p
        elif self.cfg.top_p is not None:
            request_data["top_p"] = self.cfg.top_p

        if params.stop:
            request_data["stop"] = params.stop

        if params.seed is not None:
            request_data["seed"] = params.seed

        if params.response_format:
            request_data["response_format"] = {"type": params.response_format}

        # Add extra parameters
        request_data.update(params.extra_params)
        request_data.update(self.cfg.extra_params)
        params.trace_context["llm_request_body"] = request_data.copy()

        return request_data

    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers."""
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        return headers

    async def _process_streaming_response(self, response: httpx.Response) -> AsyncGenerator[StreamingModelResponse, None]:
        """Process streaming response from API."""
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    if chunk.get("choices"):
                        choice_data = chunk["choices"][0]
                        delta = choice_data.get("delta", {})
                        
                        # Convert delta dict to Message object
                        delta_message = Message.from_openai({"role": "assistant", **delta}) if delta else Message(role="assistant", content="")
                        
                        choice = StreamingChoice(
                            index=choice_data.get("index", 0),
                            delta=delta_message,
                            finish_reason=choice_data.get("finish_reason")
                        )
                        
                        usage = None
                        if "usage" in chunk:
                            usage_data = chunk["usage"]
                            usage = Usage(
                                prompt_tokens=usage_data.get("prompt_tokens", 0),
                                completion_tokens=usage_data.get("completion_tokens", 0),
                                total_tokens=usage_data.get("total_tokens", 0)
                            )
                        
                        yield StreamingModelResponse(
                            id=chunk.get("id", f"chatcmpl-{int(time.time())}"),
                            object=chunk.get("object", "chat.completion.chunk"),
                            created=chunk.get("created", int(time.time())),
                            model=chunk.get("model", self.cfg.get_actual_model_name()),
                            choices=[choice],
                            usage=usage
                        )
                except json.JSONDecodeError:
                    continue

    def _process_non_streaming_response(self, response: httpx.Response, trace_context: dict = None) -> ModelResponse:
        """Process non-streaming response from API."""
        data = response.json()
        
        # Record original response in trace context
        if trace_context is not None:
            trace_context["llm_response_body"] = data.copy()
        
        choices = []
        for choice_data in data.get("choices", []):
            choice = Choice(
                index=choice_data.get("index", 0),
                message=choice_data.get("message", {}),
                finish_reason=choice_data.get("finish_reason")
            )
            choices.append(choice)
        
        usage = None
        if "usage" in data:
            usage_data = data["usage"]
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0)
            )
        
        return ModelResponse(
            id=data.get("id", f"chatcmpl-{int(time.time())}"),
            object=data.get("object", "chat.completion"),
            created=data.get("created", int(time.time())),
            model=data.get("model", self.cfg.get_actual_model_name()),
            choices=choices,
            usage=usage
        )


    def _create_error_response(self, error_message: str, is_streaming: bool = False,
                              error_type: str | None = None, error_code: str | None = None) -> Union[ModelResponse, StreamingModelResponse]:
        """Create error response."""
        # Check if this is a context length error
        if is_context_length_error(error_message):
            error_object = {
                "message": error_message,
                "type": "context_length_exceed_error",
                "code": "context_length_exceed"
            }
        else:
            # Create error object for the error field
            error_object = {
                "message": error_message,
                "type": error_type or "pseudo_function_calling_error",
                "code": error_code or "request_error"
            }

        if is_streaming:
            return StreamingModelResponse(
                id=f"error-{int(time.time())}",
                object="chat.completion.chunk",
                created=int(time.time()),
                model=self.cfg.get_actual_model_name(),
                choices=[],
                usage=None,
                error=error_object
            )
        else:
            return ModelResponse(
                id=f"error-{int(time.time())}",
                object="chat.completion",
                created=int(time.time()),
                model=self.cfg.get_actual_model_name(),
                choices=[],
                usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                error=error_object
            )


class SyncPseudoFunctionCallingClient(SyncModelClient):
    """Synchronous version of the pseudo function calling client."""

    provider = "pseudo_function_calling"

    def __init__(self, cfg: ModelConfig, client: httpx.Client | None = None, is_debug: bool = False) -> None:
        """Initialize the sync client."""
        super().__init__(cfg, client, is_debug=is_debug)

    def _generate_curl_command(self, url: str, headers: Dict[str, str], request_data: Dict[str, Any]) -> str:
        """Generate equivalent curl command for debugging."""
        curl_parts = ["curl", "-X", "POST"]
        
        # Add headers
        for key, value in headers.items():
            curl_parts.extend(["-H", f"{key}: {value}"])
        
        # Add JSON data
        json_data = json.dumps(request_data, ensure_ascii=False)
        curl_parts.extend(["-d", json_data])
        
        # Add URL
        curl_parts.append(url)
        
        # Use shlex.join for proper escaping (Python 3.8+)
        try:
            return shlex.join(curl_parts)
        except AttributeError:
            # Fallback for older Python versions
            return " ".join(shlex.quote(part) for part in curl_parts)

    def _run(self, params: RunParams) -> Generator[Union[ModelResponse, StreamingModelResponse], None, None]:
        """Execute the API call with pseudo function calling support (sync version)."""
        # Initialize basic trace context - will be updated with actual call details later
        params.trace_context["responses"] = []
        
        # Check if we have tool calls to handle
        has_tools = params.tool_params and getattr(params.tool_params, 'tools', None)
        
        if not has_tools:
            # No tools, but still merge consecutive user messages
            messages = [msg.to_openai() for msg in params.messages]
            merged_messages = _merge_consecutive_user_messages(messages)
            
            # Create modified params with merged messages
            no_tools_params = copy.deepcopy(params)
            no_tools_params.messages = [Message.from_openai(msg) for msg in merged_messages]
            no_tools_params.trace_context = params.trace_context
            for response in self._make_regular_call(no_tools_params):
                yield response
                return

        # Convert messages to pseudo function calling format
        tools = [tool.model_dump() if hasattr(tool, 'model_dump') else tool 
                for tool in params.tool_params.tools]
        
        # Convert messages format
        messages = [msg.to_openai() for msg in params.messages]
        converted_messages = convert_fncall_messages_to_non_fncall_messages(messages, tools)
        
        # Create modified params for pseudo FC call
        pseudo_params = copy.deepcopy(params)
        pseudo_params.messages = [Message.from_openai(msg) for msg in converted_messages]
        pseudo_params.tool_params = None
        
        # Add stop words
        if pseudo_params.stop:
            if isinstance(pseudo_params.stop, str):
                pseudo_params.stop = [pseudo_params.stop] + STOP_WORDS
            else:
                pseudo_params.stop = list(pseudo_params.stop) + STOP_WORDS
        else:
            pseudo_params.stop = STOP_WORDS
        pseudo_params.trace_context = params.trace_context

        # Make the call and try to parse response
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                for response in self._make_regular_call(pseudo_params):
                    # Check if this is already an error response
                    if (hasattr(response, 'choices') and response.choices and 
                        response.choices[0].finish_reason == "error"):
                        # This is an error response, don't retry, just return it
                        yield response
                        return
                    
                    # Try to convert response back to function calling format
                    converted_response = self._convert_response_to_function_calling(
                        response, tools, retry_count < max_retries, params.trace_context
                    )
                    
                    if converted_response:
                        yield converted_response
                        return
                    else:
                        # Failed to parse, retry if possible
                        if retry_count < max_retries:
                            retry_count += 1
                            logger.warning(f"Failed to parse function call, retrying ({retry_count}/{max_retries})")
                            continue
                        else:
                            # Return original response if parsing fails
                            logger.warning("Failed to parse function call, returning original response")
                            yield response
                            return
                            
            except Exception as e:
                if retry_count < max_retries:
                    retry_count += 1
                    wait_time = calculate_retry_delay(retry_count - 1, error=e)
                    logger.warning(f"Request failed, retrying in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
                    continue
                else:
                    # Final error handling
                    logger.error(f"All retries failed: {str(e)}")
                    yield handle_model_client_error(e, params.stream, self._create_error_response)
                    return

    def _make_regular_call(self, params: RunParams) -> Generator[Union[ModelResponse, StreamingModelResponse], None, None]:
        """Make a regular OpenAI-compatible API call (sync version)."""
        request_data = self._build_request_data(params)
        url = self.cfg.api_url or "https://api.openai.com/v1/chat/completions"
        headers = self._build_headers()

        timeout = params.timeout if params.timeout is not None else 600
        
        # Generate and log curl command
        curl_command = self._generate_curl_command(url, headers, request_data)
        logger.info("start calling...")
        logger.debug(f"Equivalent curl command:\n{curl_command}")
        try:
            if params.stream:
                with self._client.stream("POST", url, headers=headers, json=request_data, timeout=timeout) as response:
                    response.raise_for_status()
                    for message in self._process_streaming_response(response):
                        yield message
            else:
                response = self._client.post(url=url, headers=headers, json=request_data, timeout=timeout)
                response.raise_for_status()
                yield self._process_non_streaming_response(response, params.trace_context)
        except Exception as e:
            # Handle HTTP errors and timeouts
            logger.error(f"HTTP request failed: {str(e)}")
            yield handle_model_client_error(e, params.stream, self._create_error_response)
            return


    def _convert_response_to_function_calling(self, response: Union[ModelResponse, StreamingModelResponse], 
                                           tools: list[dict], allow_retry: bool = True, trace_context: dict = None) -> Optional[Union[ModelResponse, StreamingModelResponse]]:
        """Convert pseudo function calling response back to standard format (sync version)."""
        if isinstance(response, StreamingModelResponse):
            logger.warning("Streaming not fully supported with pseudo function calling conversion")
            return response
            
        if not (hasattr(response, 'choices') and response.choices and response.choices[0].message):
            return response

        try:
            logger.info(response)
            content = response.choices[0].message.content or ""
            
            # Record original pseudo function calling response before conversion
            if trace_context is not None:
                original_pseudo_fc_data = {
                    "id": response.id,
                    "object": response.object,
                    "created": response.created,
                    "model": response.model,
                    "choices": [{
                        "index": choice.index,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content,
                            "tool_calls": None  # No tool calls in pseudo FC response
                        },
                        "finish_reason": choice.finish_reason
                    } for choice in response.choices],
                    "usage": response.usage.model_dump() if hasattr(response.usage, 'model_dump') else response.usage.__dict__ if response.usage else None
                }
                trace_context["response_extra"] = {"raw": original_pseudo_fc_data}
            
            # Try to parse function call
            temp_messages = [{"role": "assistant", "content": content}]
            converted_back, parse_error = convert_non_fncall_messages_to_fncall_messages(temp_messages, tools)
            
            if parse_error:
                # Parsing failed - return error response instead of retrying
                logger.error(f"Tool call parsing failed: {parse_error}")
                error_response = self._create_error_response(parse_error, False)
                return error_response
            
            if converted_back and converted_back[0].get("tool_calls"):
                # Successfully parsed tool calls
                tool_calls = converted_back[0]["tool_calls"]
                response.choices[0].message.tool_calls = tool_calls
                response.choices[0].message.content = converted_back[0]["content"]
                return response
            else:
                # No tool calls found
                return None if allow_retry else response
                
        except Exception as e:
            logger.error(f"Failed to convert pseudo FC response: {str(e)}")
            # Return error response instead of original response to ensure proper error handling
            error_response = self._create_error_response(f"Response parsing failed: {str(e)}", False)
            return error_response

    def _build_request_data(self, params: RunParams) -> Dict[str, Any]:
        """Build request data for OpenAI-compatible API."""
        messages = []
        for m in params.messages:
            openai_msg = m.to_openai()
            if 'content' in openai_msg and isinstance(openai_msg['content'], list):
                openai_msg['content'] = convert_image_urls_to_base64(openai_msg['content'])
            messages.append(openai_msg)

        request_data = {
            "model": self.cfg.get_actual_model_name(),
            "messages": messages,
            "stream": params.stream,
        }

        # Add parameters if set
        if params.temperature is not None:
            request_data["temperature"] = params.temperature
        elif self.cfg.temperature is not None:
            request_data["temperature"] = self.cfg.temperature

        if params.max_tokens is not None:
            request_data["max_tokens"] = params.max_tokens
        elif self.cfg.max_tokens is not None:
            request_data["max_tokens"] = self.cfg.max_tokens

        if params.top_p is not None:
            request_data["top_p"] = params.top_p
        elif self.cfg.top_p is not None:
            request_data["top_p"] = self.cfg.top_p

        if params.stop:
            request_data["stop"] = params.stop

        if params.seed is not None:
            request_data["seed"] = params.seed

        if params.response_format:
            request_data["response_format"] = {"type": params.response_format}

        # Add extra parameters
        request_data.update(params.extra_params)
        request_data.update(self.cfg.extra_params)
        params.trace_context["llm_request_body"] = request_data.copy()
        return request_data

    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers."""
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        return headers

    def _process_streaming_response(self, response: httpx.Response) -> Generator[StreamingModelResponse, None, None]:
        """Process streaming response from API (sync version)."""
        for line in response.iter_lines():
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    if chunk.get("choices"):
                        choice_data = chunk["choices"][0]
                        delta = choice_data.get("delta", {})
                        
                        # Convert delta dict to Message object
                        delta_message = Message.from_openai({"role": "assistant", **delta}) if delta else Message(role="assistant", content="")
                        
                        choice = StreamingChoice(
                            index=choice_data.get("index", 0),
                            delta=delta_message,
                            finish_reason=choice_data.get("finish_reason")
                        )
                        
                        usage = None
                        if "usage" in chunk:
                            usage_data = chunk["usage"]
                            usage = Usage(
                                prompt_tokens=usage_data.get("prompt_tokens", 0),
                                completion_tokens=usage_data.get("completion_tokens", 0),
                                total_tokens=usage_data.get("total_tokens", 0)
                            )
                        
                        yield StreamingModelResponse(
                            id=chunk.get("id", f"chatcmpl-{int(time.time())}"),
                            object=chunk.get("object", "chat.completion.chunk"),
                            created=chunk.get("created", int(time.time())),
                            model=chunk.get("model", self.cfg.get_actual_model_name()),
                            choices=[choice],
                            usage=usage
                        )
                except json.JSONDecodeError:
                    continue

    def _process_non_streaming_response(self, response: httpx.Response, trace_context: dict = None) -> ModelResponse:
        """Process non-streaming response from API."""
        data = response.json()
        
        # Record original response in trace context
        if trace_context is not None:
            trace_context["llm_response_body"] = data.copy()
        
        choices = []
        for choice_data in data.get("choices", []):
            choice = Choice(
                index=choice_data.get("index", 0),
                message=choice_data.get("message", {}),
                finish_reason=choice_data.get("finish_reason")
            )
            choices.append(choice)
        
        usage = None
        if "usage" in data:
            usage_data = data["usage"]
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0)
            )
        
        return ModelResponse(
            id=data.get("id", f"chatcmpl-{int(time.time())}"),
            object=data.get("object", "chat.completion"),
            created=data.get("created", int(time.time())),
            model=data.get("model", self.cfg.get_actual_model_name()),
            choices=choices,
            usage=usage
        )


    def _create_error_response(self, error_message: str, is_streaming: bool = False,
                              error_type: str | None = None, error_code: str | None = None) -> Union[ModelResponse, StreamingModelResponse]:
        """Create error response."""
        # Check if this is a context length error
        if is_context_length_error(error_message):
            error_object = {
                "message": error_message,
                "type": "context_length_exceed_error",
                "code": "context_length_exceed"
            }
        else:
            # Create error object for the error field
            error_object = {
                "message": error_message,
                "type": error_type or "pseudo_function_calling_error",
                "code": error_code or "request_error"
            }

        if is_streaming:
            return StreamingModelResponse(
                id=f"error-{int(time.time())}",
                object="chat.completion.chunk",
                created=int(time.time()),
                model=self.cfg.get_actual_model_name(),
                choices=[],
                usage=None,
                error=error_object
            )
        else:
            return ModelResponse(
                id=f"error-{int(time.time())}",
                object="chat.completion",
                created=int(time.time()),
                model=self.cfg.get_actual_model_name(),
                choices=[],
                usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                error=error_object
            )