"""Image utility functions for OpenAI clients."""

import base64
import hashlib
import mimetypes
from typing import Dict, Any, List, Union, Optional
import httpx
import threading

from ..logger import get_logger

logger = get_logger(__name__)


# Global mapping storage for base64 hash to original URL
# 使用线程锁确保线程安全
_url_mapping_lock = threading.Lock()
_base64_hash_to_url_mapping: Dict[str, str] = {}


def _compute_base64_hash(base64_data_url: str) -> str:
    """计算base64 data URL的hash值作为key
    
    Args:
        base64_data_url: base64格式的data URL
        
    Returns:
        base64内容的MD5 hash值
    """
    try:
        # 提取base64数据部分（去掉data:image/xxx;base64,前缀）
        if ',' in base64_data_url:
            base64_content = base64_data_url.split(',', 1)[1]
        else:
            base64_content = base64_data_url
        
        # 计算MD5 hash
        return hashlib.md5(base64_content.encode('utf-8')).hexdigest()
    except Exception as e:
        logger.warning(f"Failed to compute hash for base64 data: {e}")
        return ""


def store_url_mapping(base64_data_url: str, original_url: str) -> str:
    """存储base64 hash到原始URL的映射
    
    Args:
        base64_data_url: base64格式的data URL
        original_url: 原始图片URL
        
    Returns:
        base64的hash key
    """
    hash_key = _compute_base64_hash(base64_data_url)
    if hash_key:
        with _url_mapping_lock:
            _base64_hash_to_url_mapping[hash_key] = original_url
            logger.debug(f"Stored URL mapping: {hash_key} -> {original_url}")
    return hash_key


def get_original_url_by_hash(hash_key: str) -> Optional[str]:
    """根据hash key获取原始URL
    
    Args:
        hash_key: base64的hash key
        
    Returns:
        原始URL，如果找不到则返回None
    """
    with _url_mapping_lock:
        return _base64_hash_to_url_mapping.get(hash_key)


def clear_url_mapping():
    """清空URL映射（用于测试或内存管理）"""
    with _url_mapping_lock:
        _base64_hash_to_url_mapping.clear()
        logger.debug("Cleared URL mapping storage")


def fetch_image_as_base64(image_url: str, timeout: float = 30.0) -> str:
    """Download image from URL and convert to base64 data URL.
    
    Args:
        image_url: The URL of the image to download
        timeout: Request timeout in seconds
        
    Returns:
        Base64 data URL string (e.g., "data:image/png;base64,...")
        
    Raises:
        httpx.HTTPError: If the request fails
        ValueError: If the URL is invalid
    """
    if not image_url or not isinstance(image_url, str):
        raise ValueError("Invalid image URL")
    
    # Skip if already a base64 data URL
    if image_url.startswith("data:"):
        logger.info(f"Image already in base64 format, skipping conversion")
        return image_url
    
    logger.info(f"Converting image URL to base64: {image_url}")
    
    with httpx.Client(timeout=timeout) as client:
        response = client.get(image_url)
        response.raise_for_status()
        
        # Get content type from response or guess from URL
        content_type = response.headers.get("content-type")
        logger.info(f"Original content-type from server: {content_type}")
        
        # If content-type is application/octet-stream or not provided, try to guess from URL
        if not content_type or content_type == "application/octet-stream":
            guessed_type, _ = mimetypes.guess_type(image_url)
            if guessed_type and guessed_type.startswith("image/"):
                content_type = guessed_type
            else:
                # Try to detect from file extension more aggressively
                url_lower = image_url.lower()
                if url_lower.endswith(('.jpg', '.jpeg')):
                    content_type = "image/jpeg"
                elif url_lower.endswith('.png'):
                    content_type = "image/png"
                elif url_lower.endswith('.gif'):
                    content_type = "image/gif"
                elif url_lower.endswith('.webp'):
                    content_type = "image/webp"
                else:
                    content_type = "image/png"  # Default fallback
        
        # Additional validation: ensure content_type is an image type
        if not content_type.startswith("image/"):
            # Try to guess from URL as fallback
            guessed_type, _ = mimetypes.guess_type(image_url)
            if guessed_type and guessed_type.startswith("image/"):
                content_type = guessed_type
            else:
                # Aggressive file extension detection
                url_lower = image_url.lower()
                if url_lower.endswith(('.jpg', '.jpeg')):
                    content_type = "image/jpeg"
                elif url_lower.endswith('.png'):
                    content_type = "image/png"
                elif url_lower.endswith('.gif'):
                    content_type = "image/gif"
                elif url_lower.endswith('.webp'):
                    content_type = "image/webp"
                else:
                    content_type = "image/png"  # Default fallback
        
        logger.info(f"Final content-type determined: {content_type}")
        
        # Encode image data to base64
        b64_str = base64.b64encode(response.content).decode("utf-8")
        data_url = f"data:{content_type};base64,{b64_str}"
        
        # Store mapping from base64 hash to original URL for trace reporting
        store_url_mapping(data_url, image_url)
        
        logger.info(f"Successfully converted image to base64 data URL (size: {len(response.content)} bytes)")
        return data_url


def convert_image_urls_to_base64(content: Union[str, List[Dict[str, Any]]], timeout: float = 30.0, skip_miaoda_files: bool = True) -> Union[str, List[Dict[str, Any]]]:
    """Convert image URLs in message content to base64 data URLs.
    
    Args:
        content: Message content (string or list of content objects)
        timeout: Request timeout in seconds
        skip_miaoda_files: If True, skip conversion for miaoda-conversation-file URLs. Default is True.
        
    Returns:
        Content with image URLs converted to base64 data URLs
    """
    if isinstance(content, str):
        return content
    
    if not isinstance(content, list):
        return content
    
    converted_content = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "image_url":
            image_url_obj = item.get("image_url", {})
            if isinstance(image_url_obj, dict):
                url = image_url_obj.get("url", "")
                should_skip = url.startswith("data:")
                if skip_miaoda_files and url.startswith("https://miaoda-conversation-file"):
                    should_skip = True
                
                if url and not should_skip:
                    try:
                        logger.info(f"Converting image URL in message content: {url}")
                        base64_url = fetch_image_as_base64(url, timeout)
                        new_item = item.copy()
                        new_item["image_url"] = image_url_obj.copy()
                        new_item["image_url"]["url"] = base64_url
                        converted_content.append(new_item)
                        logger.info(f"Successfully converted image URL to base64 in message content")
                    except Exception as e:
                        logger.warning(f"Failed to convert image URL to base64: {e}. Keeping original URL.")
                        converted_content.append(item)
                else:
                    converted_content.append(item)
            else:
                converted_content.append(item)
        else:
            converted_content.append(item)
    
    return converted_content


def preserve_original_image_urls(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create a copy of messages with original image URLs preserved for reporting.
    
    Args:
        messages: List of OpenAI format messages
        
    Returns:
        Deep copy of messages with original URLs preserved
    """
    import copy
    return copy.deepcopy(messages)


