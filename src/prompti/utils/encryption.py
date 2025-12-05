"""
配置加密/解密工具

用于解密从 PromptStore 获取的加密配置
"""

import os
import base64
from typing import Optional

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import padding
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

from ..logger import get_logger
logger = get_logger(__name__)

# 默认加密密钥 (与 PromptStore 保持一致)
DEFAULT_ENCRYPTION_KEY = "agentos-promptstore-default-encryption-key-change-in-production"


def get_encryption_key() -> Optional[bytes]:
    """
    从环境变量获取加密密钥,如果未设置则使用默认密钥

    优先级:
    1. PROMPTI_ENCRYPTION_KEY 环境变量
    2. ENCRYPTION_KEY 环境变量
    3. 默认密钥 (与 PromptStore 一致)

    Returns:
        32 字节密钥
    """
    key = os.getenv("PROMPTI_ENCRYPTION_KEY") or os.getenv("ENCRYPTION_KEY") or DEFAULT_ENCRYPTION_KEY

    if key == DEFAULT_ENCRYPTION_KEY:
        logger.debug("Using default ENCRYPTION_KEY (matches PromptStore default)")

    # 确保密钥长度正确 (AES-256 需要 32 字节)
    key_bytes = key.encode('utf-8')
    if len(key_bytes) != 32:
        # 使用 SHA-256 哈希生成固定长度密钥
        import hashlib
        key_bytes = hashlib.sha256(key_bytes).digest()

    return key_bytes


def decrypt_aes256(encrypted_base64: str, key: Optional[bytes] = None) -> Optional[str]:
    """
    使用 AES-256-CBC 解密

    Args:
        encrypted_base64: Base64 编码的加密字符串
        key: 32 字节加密密钥,如果为 None 则从环境变量获取

    Returns:
        明文字符串,解密失败返回 None
    """
    if not CRYPTO_AVAILABLE:
        logger.warning("cryptography library not installed, cannot decrypt config")
        return None

    try:
        # 获取密钥
        if key is None:
            key = get_encryption_key()
            if key is None:
                logger.warning("Encryption key not found in environment variables")
                return None

        # Base64 解码
        encrypted = base64.b64decode(encrypted_base64)

        # 提取 IV 和密文
        iv = encrypted[:16]
        ciphertext = encrypted[16:]

        # 创建解密器
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()

        # 解密
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()

        # 移除填充
        unpadder = padding.PKCS7(128).unpadder()
        plaintext = unpadder.update(padded_data) + unpadder.finalize()

        return plaintext.decode('utf-8')

    except Exception as e:
        logger.error(f"Failed to decrypt config: {e}")
        return None


def decrypt_config_field(config: dict, field_path: str, key: Optional[bytes] = None) -> dict:
    """
    解密配置中的指定字段

    Args:
        config: 配置字典
        field_path: 字段路径,如 "redis.password"
        key: 加密密钥

    Returns:
        解密后的配置字典 (原地修改)

    Example:
        config = {
            "redis": {
                "password": "***"
            }
        }
        decrypt_config_field(config, "redis.password")
        # config["redis"]["password"] = "***"
    """
    try:
        # 解析字段路径
        parts = field_path.split('.')
        current = config

        # 定位到字段
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                return config
            current = current[part]

        field_name = parts[-1]
        if field_name not in current:
            return config

        # 解密
        encrypted_value = current[field_name]
        if not encrypted_value or not isinstance(encrypted_value, str):
            return config

        decrypted = decrypt_aes256(encrypted_value, key)
        if decrypted is not None:
            current[field_name] = decrypted
            logger.debug(f"Successfully decrypted field: {field_path}")
        else:
            logger.warning(f"Failed to decrypt field: {field_path}")

    except Exception as e:
        logger.error(f"Error decrypting field {field_path}: {e}")

    return config


def decrypt_global_config(config: dict, key: Optional[bytes] = None) -> dict:
    """
    解密全局配置中的所有加密字段

    已知加密字段:
    - redis.password

    Args:
        config: 全局配置字典
        key: 加密密钥

    Returns:
        解密后的配置字典
    """
    # 解密 redis.password
    if "redis" in config and "password" in config["redis"]:
        decrypt_config_field(config, "redis.password", key)

    # 未来可以添加更多加密字段
    # decrypt_config_field(config, "other.secret", key)

    return config


# 便捷函数
def is_encryption_available() -> bool:
    """检查加密库是否可用"""
    return CRYPTO_AVAILABLE
