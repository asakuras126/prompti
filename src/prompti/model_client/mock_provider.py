"""Mock Provider - Sequential mock client as a standard provider."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Union, Generator, Optional, AsyncGenerator
import os

from .base import ModelClient, SyncModelClient, ModelConfig, RunParams
from ..message import ModelResponse, StreamingModelResponse
from ..logger import get_logger

logger = get_logger(__name__)

try:
    # Try elasticsearch first, then fallback to opensearch
    try:
        from elasticsearch import Elasticsearch
        ES_CLIENT_TYPE = "elasticsearch"
    except ImportError:
        from opensearchpy import OpenSearch as Elasticsearch
        ES_CLIENT_TYPE = "opensearch"
    ES_AVAILABLE = True
except ImportError:
    ES_AVAILABLE = False
    ES_CLIENT_TYPE = "none"


class _GlobalSequenceManager:
    """Global sequence manager for mock provider (singleton)."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
                    # Mock provider initialized
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.calls: List[Dict[str, Any]] = []
        self.call_index = 0
        self.data_lock = threading.Lock()
        self._initialized = True
        
        # Query matching related attributes
        self.query_based_data: Dict[str, List[Dict[str, Any]]] = {}
        self.current_conversation_list = None
        self.current_conversation_index = 0
        self.current_response_index = 0
        
        # Elasticsearch configuration
        self.es_client: Optional[Elasticsearch] = None
        self.es_index = "promptstore_llm_mock"
        self.use_es = False
        
        # Initialize ES if available
        self._init_elasticsearch()
        
        # Auto-load default data
        self._auto_load_data()
    
    def _init_elasticsearch(self):
        """Initialize Elasticsearch client."""
        if not ES_AVAILABLE:
            logger.warning(" Elasticsearch/OpenSearch not available, using local mock data only")
            return
            
        logger.debug(f" Using {ES_CLIENT_TYPE} client for mock data")
            
        try:
            # Get ES configuration from environment variables
            es_host = os.getenv('MOCK_ES_HOST', '10.224.55.246:8200')
            es_username = os.getenv('MOCK_ES_USERNAME', 'superuser')
            es_password = os.getenv('MOCK_ES_PASSWORD')
            es_use_ssl = os.getenv('MOCK_ES_USE_SSL', 'false').lower() == 'true'
            
            # Initialize ES client
            es_config = {
                'hosts': [es_host],
                'timeout': 30,
                'max_retries': 3,
                'retry_on_timeout': True
            }
            
            if es_username and es_password:
                es_config['http_auth'] = (es_username, es_password)
            
            if es_use_ssl:
                es_config['use_ssl'] = True
                es_config['verify_certs'] = False  # For development
            
            self.es_client = Elasticsearch(**es_config)
            
            # Test connection
            if self.es_client.ping():
                self.use_es = True
                logger.debug(f" Elasticsearch connected: {es_host}")
            else:
                logger.error(f" Elasticsearch ping failed: {es_host}")
                
        except Exception as e:
            logger.warning(f" Failed to initialize Elasticsearch: {e}")
            self.es_client = None
            self.use_es = False
    
    def _query_from_elasticsearch(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Query conversations from Elasticsearch."""
        if not self.use_es or not self.es_client:
            return None
            
        try:
            # Use match_phrase for exact phrase matching
            search_body = {
                "query": {
                    "match_phrase": {
                        "query": query
                    }
                },
                "_source": ["conversations", "total_conversations"],
                "size": 1
            }
            
            response = self.es_client.search(
                index=self.es_index,
                body=search_body
            )
            
            if response['hits']['total']['value'] > 0:
                hit = response['hits']['hits'][0]
                conversations = hit['_source'].get('conversations', [])
                logger.debug(f" Found {len(conversations)} conversations in ES for query: '{query}'")
                return conversations
            else:
                logger.debug(f" No conversations found in ES for query: '{query}'")
                return None
                
        except Exception as e:
            logger.warning(f" Elasticsearch query failed: {e}")
            return None
    
    def _auto_load_data(self):
        """Auto-load mock data from default locations."""
        # Try multiple locations for mock data
        potential_files = [
            Path(__file__).parent / "mock_datas" / "mock_data.json"
        ]
        
        for file_path in potential_files:
            file_path = Path(file_path)
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        raw_data = json.load(f)
                    
                    # Check if it's the new query-based format (dict with query keys)
                    if isinstance(raw_data, dict):
                        self.query_based_data = raw_data
                        # Also convert to old format for backward compatibility
                        self.calls = []
                        for query, conversation_list in raw_data.items():
                            for conversation in conversation_list:
                                self.calls.append(conversation)
                        logger.debug(f" Mock provider auto-loaded {len(self.query_based_data)} query-based conversations from {file_path}")
                    else:
                        # Old format (list of calls)
                        self.calls = raw_data
                        logger.debug(f" Mock provider auto-loaded {len(self.calls)} calls from {file_path}")
                    return
                except Exception as e:
                    logger.warning(f" Failed to load {file_path}: {e}")
                    continue

        logger.debug(f" Mock provider using fallback data: {len(self.calls)} calls")
    
    def _extract_user_query(self, params: RunParams) -> str:
        """Extract user query from RunParams messages."""
        if not hasattr(params, 'messages') or not params.messages:
            return ""
        
        # Find the first user message
        for message in params.messages:
            if message.role == 'user':
                content = message.content
                if isinstance(content, list):
                    # Extract text from content list
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            return item.get('text', '').strip()
                elif isinstance(content, str):
                    return content.strip()
        return ""
    
    def _find_matching_conversation(self, query: str) -> List[Dict[str, Any]]:
        """Find conversation data list that matches the user query."""
        if not query:
            return None
        
        # First try Elasticsearch
        es_conversations = self._query_from_elasticsearch(query)
        if es_conversations:
            logger.debug(f" Using ES conversations for query: '{query}'")
            return es_conversations
            
        # Fallback to local data - First try direct query match in new format
        if query in self.query_based_data:
            logger.debug(f" Direct query match found in local data for: '{query}'")
            return self.query_based_data[query]
        
        # Fallback to old matching logic for backward compatibility
        for call_data in self.calls:
            params = call_data.get('params', {})
            messages = params.get('messages', [])
            
            # Find the first user message in this conversation
            for message in messages:
                if message.get('role') == 'user':
                    content = message.get('content', '')
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                if item.get('text', '').strip() == query:
                                    logger.debug(f" Using local fallback data for query: '{query}'")
                                    return [call_data]
                    elif isinstance(content, str) and content.strip() == query:
                        logger.debug(f" Using local fallback data for query: '{query}'")
                        return [call_data]
                    break  # Only check first user message
        return None
    

    def load_custom_data(self, data_source: Union[str, Path, List[Dict[str, Any]], Dict[str, Any]]):
        """Load custom mock data."""
        with self.data_lock:
            if isinstance(data_source, (str, Path)):
                data_path = Path(data_source)
                if not data_path.exists():
                    raise FileNotFoundError(f"Mock data file not found: {data_path}")
                
                with open(data_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                
                # Check if it's the new query-based format
                if isinstance(raw_data, dict) and not any(key in raw_data for key in ['params', 'responses']):
                    self.query_based_data = raw_data
                    # Convert to old format for backward compatibility
                    self.calls = []
                    for query, conversation_list in raw_data.items():
                        for conversation in conversation_list:
                            self.calls.append(conversation)
                    logger.debug(f" Mock provider loaded {len(self.query_based_data)} query-based conversations from {data_path}")
                else:
                    # Old format
                    self.calls = raw_data if isinstance(raw_data, list) else [raw_data]
                    logger.debug(f" Mock provider loaded {len(self.calls)} calls from {data_path}")
                
            elif isinstance(data_source, dict):
                # Direct dict input - check format
                if not any(key in data_source for key in ['params', 'responses']):
                    # Query-based format
                    self.query_based_data = data_source
                    self.calls = []
                    for query, conversation_list in data_source.items():
                        for conversation in conversation_list:
                            self.calls.append(conversation)
                    logger.debug(f" Mock provider loaded {len(self.query_based_data)} query-based conversations from provided dict")
                else:
                    # Single call format
                    self.calls = [data_source]
                    logger.debug(f" Mock provider loaded 1 call from provided dict")
                
            elif isinstance(data_source, list):
                self.calls = data_source.copy()
                logger.debug(f" Mock provider loaded {len(self.calls)} calls from provided data")
                
            self.call_index = 0

    def _extract_wordspace_dir(self, params: RunParams = None) -> str:
        messages = params.messages
        workspace_dir = ""
        try:
            message_item = messages[2]
            content = message_item.content
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text = item.get('text', '')
                        #  REPOSITORY_INFO  repo_dir 
                        import re
                        match = re.search(r'repo_dir:\s*([^\s\n.]+)', text)
                        if match:
                            workspace_dir = match.group(1).strip()
                            break
            elif isinstance(content, str):
                #  REPOSITORY_INFO  repo_dir 
                import re
                match = re.search(r'repo_dir:\s*([^\s\n.]+)', content)
                if match:
                    workspace_dir = match.group(1).strip()
        except Exception as e:
            pass
        return workspace_dir

    def _replace_workspace_paths(self, responses: List[Dict[str, Any]], workspace_dir: str) -> List[Dict[str, Any]]:
        """Replace /workspace/app-*** paths with the extracted workspace directory."""
        if not workspace_dir:
            return responses
            
        import re
        import json
        
        #  /workspace/app-*** 
        workspace_pattern = r'/workspace/app-[a-zA-Z0-9_-]+'
        
        replaced_responses = []
        for response in responses:
            # 
            response_str = json.dumps(response)
            # workspace
            response_str = re.sub(workspace_pattern, workspace_dir, response_str)
            replaced_response = json.loads(response_str)
            replaced_responses.append(replaced_response)
            
        return replaced_responses

    def _check_special_error_queries(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Check if query matches special error conditions and return appropriate error responses."""
        if not query:
            return None
            
        query_lower = query.lower().strip()
        
        # Context exceed error
        if "context exceed" in query_lower:
            return [{
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant", 
                        "content": "[ERROR] context_length_exceed_error: The input context is too long and exceeds the maximum allowed length."
                    }
                }],
                "error": {
                    "type": "context_length_exceed_error",
                    "code": "context_length_exceed",
                    "message": "Context length exceeded maximum limit"
                }
            }]
            
        # Timeout error  
        if "timeout error" in query_lower:
            return [{
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "[ERROR] timeout_error: Request timed out while processing."
                    }
                }],
                "error": {
                    "type": "timeout_error",
                    "code": "request_timeout", 
                    "message": "Request timeout"
                }
            }]
            
        # Rate limit error
        if "rate limit" in query_lower:
            return [{
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "[ERROR] rate_limit_error: API rate limit exceeded. Please try again later."
                    }
                }],
                "error": {
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                    "message": "Rate limit exceeded"
                }
            }]
            
        # Authentication error
        if "auth error" in query_lower or "authentication" in query_lower:
            return [{
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "[ERROR] authentication_error: Authentication failed. Please check your credentials."
                    }
                }],
                "error": {
                    "type": "authentication_error",
                    "code": "auth_failed",
                    "message": "Authentication failed"
                }
            }]
            
        return None

    def get_next_responses(self, params: RunParams = None) -> List[Dict[str, Any]]:
        """Get next responses based on query matching or sequence."""
        with self.data_lock:
            # Extract workspace directory for path replacement
            workspace_dir = self._extract_wordspace_dir(params) if params else ""
            
            # If we have current conversation list, continue with it
            if self.current_conversation_list is not None:
                logger.debug(f" Mock provider: continuing with current conversation list")
                if self.current_conversation_index < len(self.current_conversation_list):
                    current_conversation = self.current_conversation_list[self.current_conversation_index]
                    responses = current_conversation.get("responses", [])
                    
                    if self.current_response_index < len(responses):
                        # Return next response from current conversation
                        response = [responses[self.current_response_index]]
                        self.current_response_index += 1
                        logger.debug(f" Mock provider: conversation {self.current_conversation_index + 1}/{len(self.current_conversation_list)}, response {self.current_response_index}/{len(responses)}")
                        return self._replace_workspace_paths(response, workspace_dir)
                    else:
                        # Current conversation responses exhausted, move to next conversation
                        self.current_conversation_index += 1
                        self.current_response_index = 0
                        logger.debug(f" Mock provider: moving to next conversation in the chain")
                        
                        if self.current_conversation_index < len(self.current_conversation_list):
                            # Start next conversation
                            current_conversation = self.current_conversation_list[self.current_conversation_index]
                            responses = current_conversation.get("responses", [])
                            if responses:
                                response = [responses[0]]
                                self.current_response_index = 1
                                logger.debug(f" Mock provider: conversation {self.current_conversation_index + 1}/{len(self.current_conversation_list)}, response 1/{len(responses)}")
                                return self._replace_workspace_paths(response, workspace_dir)
                else:
                    # All conversations in the list exhausted, reset for new matching
                    logger.debug(f" Mock provider: all conversations completed, resetting for new query matching")
                    self.current_conversation_list = None
                    self.current_conversation_index = 0
                    self.current_response_index = 0
            
            # Try to find matching conversation based on query
            if params is not None:
                query = self._extract_user_query(params)
                if query:
                    # Check for special error queries first
                    error_response = self._check_special_error_queries(query)
                    if error_response:
                        logger.error(f" Mock provider: returning error response for query '{query}'")
                        return self._replace_workspace_paths(error_response, workspace_dir)
                    
                    matched_conversations = self._find_matching_conversation(query)
                    if matched_conversations:
                        self.current_conversation_list = matched_conversations
                        self.current_conversation_index = 0
                        self.current_response_index = 0
                        
                        # Start first conversation
                        current_conversation = matched_conversations[0]
                        responses = current_conversation.get("responses", [])
                        if responses:
                            response = [responses[0]]
                            self.current_response_index = 1
                            logger.debug(f" Mock provider: found matching conversations for query '{query}', starting conversation 1/{len(matched_conversations)}, response 1/{len(responses)}")
                            return self._replace_workspace_paths(response, workspace_dir)
                    else:
                        # No matching conversation found, raise error
                        raise ValueError(f" Mock provider: No matching conversation found for query '{query}'. Available queries: {list(self.query_based_data.keys()) if self.query_based_data else 'None'}")
                else:
                    # No query extracted, raise error
                    raise ValueError(" Mock provider: No user query found in the request parameters")
            else:
                # No params provided, raise error
                raise ValueError(" Mock provider: No parameters provided for query matching")
    
    def reset(self):
        """Reset sequence to beginning."""
        with self.data_lock:
            self.call_index = 0
            self.current_conversation_list = None
            self.current_conversation_index = 0
            self.current_response_index = 0
            logger.debug("Mock provider: sequence and query matching reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sequence statistics."""
        with self.data_lock:
            stats = {
                "total_calls": len(self.calls),
                "current_index": self.call_index,
                "remaining_calls": len(self.calls) - self.call_index if self.call_index < len(self.calls) else 0,
                "progress_percentage": (self.call_index / len(self.calls) * 100) if self.calls else 0,
                "query_based_conversations": len(self.query_based_data),
                "current_conversation_active": self.current_conversation_list is not None
            }
            
            if self.current_conversation_list:
                stats.update({
                    "current_conversation_index": self.current_conversation_index,
                    "current_response_index": self.current_response_index,
                    "total_conversations_in_chain": len(self.current_conversation_list)
                })
            
            return stats


class MockClient(ModelClient):
    """Mock client as a standard provider."""
    
    provider = "mock"  # This registers it as the 'mock' provider
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, cfg: ModelConfig, client=None, is_debug: bool = False):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, cfg: ModelConfig, client=None, is_debug: bool = False):
        """Initialize mock client.
        
        Args:
            cfg: Model configuration (mock provider config)
            client: Ignored for mock provider
            is_debug: Debug mode flag
        """
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.cfg = cfg
        self.is_debug = is_debug
        self.manager = _GlobalSequenceManager()
        
        # Check if custom mock data is specified in config
        if hasattr(cfg, 'api_url') and cfg.api_url:
            # Use api_url as mock data file path
            try:
                self.manager.load_custom_data(cfg.api_url)
            except FileNotFoundError:
                logger.warning(f" Mock data file not found: {cfg.api_url}, using default data")
        
        self._initialized = True
        logger.debug(f" MockClient created for provider '{self.provider}'")
    
    async def arun(self, params: RunParams) -> AsyncGenerator[Union[ModelResponse, StreamingModelResponse], None]:
        """Async run - return sequential mock responses."""
        try:
            import asyncio
            await asyncio.sleep(1)
            responses = self.manager.get_next_responses(params)

            for response_data in responses:
                # Fix and reconstruct response object
                if "choices" in response_data and any("delta" in choice for choice in response_data.get("choices", [])):
                    fixed_data = self._fix_streaming_response(response_data)
                    response = StreamingModelResponse(**fixed_data)
                else:
                    response = ModelResponse(**response_data)
                yield response

        except Exception as e:
            logger.warning(f" MockClient error: {e}")
            # Return fallback response
            fallback_response = StreamingModelResponse(
                choices=[{
                    "index": 0,
                    "delta": {"role": "assistant",
                               "content": f"[Mock response - sequence position {self.manager.call_index}]"}
                }]
            )
            yield fallback_response
    
    def run(self, params: RunParams) -> Generator[Union[ModelResponse, StreamingModelResponse], None, None]:
        """Sync run - return sequential mock responses."""
        try:
            import time
            time.sleep(1)
            responses = self.manager.get_next_responses(params)
            
            for response_data in responses:
                # Fix and reconstruct response object
                if "choices" in response_data and any("delta" in choice for choice in response_data.get("choices", [])):
                    fixed_data = self._fix_streaming_response(response_data)
                    response = StreamingModelResponse(**fixed_data)
                else:
                    response = ModelResponse(**response_data)
                yield response
                
        except Exception as e:
            logger.warning(f" MockClient error: {e}")
            # Return fallback response
            fallback_response = StreamingModelResponse(
                choices=[{
                    "index": 0,
                    "delta": {"role": "assistant", "content": f"[Mock response - sequence position {self.manager.call_index}]"}
                }]
            )
            yield fallback_response
    
    def _fix_streaming_response(self, response_data: dict) -> dict:
        """Fix streaming response structure."""
        fixed_data = response_data.copy()
        
        if "choices" in fixed_data:
            fixed_choices = []
            for i, choice in enumerate(fixed_data["choices"]):
                fixed_choice = choice.copy()
                
                if "index" not in fixed_choice:
                    fixed_choice["index"] = i
                    
                if "delta" in fixed_choice:
                    delta = fixed_choice["delta"].copy()
                    if "role" not in delta:
                        delta["role"] = "assistant"
                    fixed_choice["delta"] = delta
                    
                fixed_choices.append(fixed_choice)
                
            fixed_data["choices"] = fixed_choices
            
        return fixed_data
    
    async def aclose(self):
        """Close async client."""
        pass
    
    def close(self):
        """Close sync client."""
        pass


class SyncMockClient(SyncModelClient):
    """Sync mock client as a standard provider."""
    
    provider = "mock"  # This registers it as the 'mock' provider for sync
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, cfg: ModelConfig, client=None, is_debug: bool = False):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, cfg: ModelConfig, client=None, is_debug: bool = False):
        """Initialize sync mock client."""
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.cfg = cfg
        self.is_debug = is_debug
        self.manager = _GlobalSequenceManager()
        
        # Check if custom mock data is specified in config
        if hasattr(cfg, 'api_url') and cfg.api_url:
            try:
                self.manager.load_custom_data(cfg.api_url)
            except FileNotFoundError:
                logger.warning(f" Mock data file not found: {cfg.api_url}, using default data")
        
        self._initialized = True
        logger.debug(f" SyncMockClient created for provider '{self.provider}'")
    
    def run(self, params: RunParams) -> Generator[Union[ModelResponse, StreamingModelResponse], None, None]:
        """Sync run - return sequential mock responses."""
        try:
            responses = self.manager.get_next_responses(params)
            
            for response_data in responses:
                # Fix and reconstruct response object
                if "choices" in response_data and any("delta" in choice for choice in response_data.get("choices", [])):
                    fixed_data = self._fix_streaming_response(response_data)
                    response = StreamingModelResponse(**fixed_data)
                else:
                    response = ModelResponse(**response_data)
                yield response
                
        except Exception as e:
            logger.warning(f" SyncMockClient error: {e}")
            import traceback
            traceback.print_exc()
            # Return fallback response
            fallback_response = StreamingModelResponse(
                choices=[{
                    "index": 0,
                    "delta": {"role": "assistant", "content": f"[Mock response - sequence position {self.manager.call_index}]"}
                }]
            )
            yield fallback_response
    
    def _fix_streaming_response(self, response_data: dict) -> dict:
        """Fix streaming response structure."""
        fixed_data = response_data.copy()
        
        if "choices" in fixed_data:
            fixed_choices = []
            for i, choice in enumerate(fixed_data["choices"]):
                fixed_choice = choice.copy()
                
                if "index" not in fixed_choice:
                    fixed_choice["index"] = i
                    
                if "delta" in fixed_choice:
                    delta = fixed_choice["delta"].copy()
                    if "role" not in delta:
                        delta["role"] = "assistant"
                    fixed_choice["delta"] = delta
                    
                fixed_choices.append(fixed_choice)
                
            fixed_data["choices"] = fixed_choices
            
        return fixed_data
    
    def close(self):
        """Close sync client."""
        pass


# Utility functions for managing mock provider

def load_mock_data(data_source: Union[str, Path, List[Dict[str, Any]]]):
    """Load custom data into mock provider."""
    manager = _GlobalSequenceManager()
    manager.load_custom_data(data_source)


def reset_mock_sequence():
    """Reset mock sequence to beginning."""
    manager = _GlobalSequenceManager()
    manager.reset()


def get_mock_stats():
    """Get mock provider statistics."""
    manager = _GlobalSequenceManager()
    return manager.get_stats()