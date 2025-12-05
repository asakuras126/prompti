"""Tests for encryption utilities."""

import os
import base64
import pytest
from unittest.mock import patch, MagicMock

from prompti.utils.encryption import (
    get_encryption_key,
    decrypt_aes256,
    decrypt_config_field,
    decrypt_global_config,
    is_encryption_available,
    DEFAULT_ENCRYPTION_KEY,
    CRYPTO_AVAILABLE,
)


class TestEncryptionAvailability:
    """Test encryption library availability."""

    def test_is_encryption_available(self):
        """Test checking if encryption is available."""
        result = is_encryption_available()
        assert isinstance(result, bool)
        assert result == CRYPTO_AVAILABLE


class TestGetEncryptionKey:
    """Test suite for get_encryption_key."""

    def test_default_key(self):
        """Test getting default encryption key."""
        with patch.dict(os.environ, {}, clear=True):
            key = get_encryption_key()
            assert len(key) == 32  # AES-256 requires 32 bytes

    def test_prompti_encryption_key_env(self):
        """Test PROMPTI_ENCRYPTION_KEY environment variable."""
        test_key = "my-custom-key-that-is-32-bytes!"
        with patch.dict(os.environ, {"PROMPTI_ENCRYPTION_KEY": test_key}):
            key = get_encryption_key()
            assert len(key) == 32

    def test_encryption_key_env(self):
        """Test ENCRYPTION_KEY environment variable."""
        test_key = "another-custom-key-32-bytes!!"
        with patch.dict(os.environ, {"ENCRYPTION_KEY": test_key}):
            key = get_encryption_key()
            assert len(key) == 32

    def test_prompti_key_takes_precedence(self):
        """Test that PROMPTI_ENCRYPTION_KEY takes precedence."""
        key1 = "prompti-key-that-is-32-bytes!"
        key2 = "encryption-key-that-is-32-byte"
        with patch.dict(os.environ, {
            "PROMPTI_ENCRYPTION_KEY": key1,
            "ENCRYPTION_KEY": key2
        }):
            key = get_encryption_key()
            # Should hash key1, not key2
            import hashlib
            expected = hashlib.sha256(key1.encode('utf-8')).digest()
            assert key == expected

    def test_short_key_hashed(self):
        """Test that short keys are hashed to 32 bytes."""
        short_key = "short"
        with patch.dict(os.environ, {"ENCRYPTION_KEY": short_key}):
            key = get_encryption_key()
            assert len(key) == 32

            # Verify it's the SHA-256 hash
            import hashlib
            expected = hashlib.sha256(short_key.encode('utf-8')).digest()
            assert key == expected

    def test_long_key_hashed(self):
        """Test that long keys are hashed to 32 bytes."""
        long_key = "a" * 100
        with patch.dict(os.environ, {"ENCRYPTION_KEY": long_key}):
            key = get_encryption_key()
            assert len(key) == 32

    def test_exactly_32_bytes_not_hashed(self):
        """Test that exactly 32-byte keys are used as-is."""
        exact_key = "a" * 32
        with patch.dict(os.environ, {"ENCRYPTION_KEY": exact_key}):
            key = get_encryption_key()
            assert len(key) == 32
            assert key == exact_key.encode('utf-8')


@pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not installed")
class TestDecryptAES256:
    """Test suite for decrypt_aes256."""

    def create_encrypted_value(self, plaintext: str, key: bytes) -> str:
        """Helper to create encrypted values for testing."""
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import padding
        import os

        # Add padding
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(plaintext.encode('utf-8')) + padder.finalize()

        # Generate IV
        iv = os.urandom(16)

        # Encrypt
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        # Combine IV and ciphertext, then base64 encode
        encrypted = base64.b64encode(iv + ciphertext).decode('utf-8')
        return encrypted

    def test_decrypt_success(self):
        """Test successful decryption."""
        plaintext = "my-secret-password"
        key = get_encryption_key()
        encrypted = self.create_encrypted_value(plaintext, key)

        result = decrypt_aes256(encrypted, key)
        assert result == plaintext

    def test_decrypt_with_default_key(self):
        """Test decryption using default key from environment."""
        plaintext = "test-secret"
        key = get_encryption_key()
        encrypted = self.create_encrypted_value(plaintext, key)

        # Don't pass key, should use environment variable
        with patch.dict(os.environ, {}, clear=True):
            result = decrypt_aes256(encrypted, key=None)
            assert result == plaintext

    def test_decrypt_invalid_base64(self):
        """Test decryption with invalid base64."""
        result = decrypt_aes256("not-valid-base64!!!", None)
        assert result is None

    def test_decrypt_wrong_key(self):
        """Test decryption with wrong key."""
        plaintext = "secret"
        correct_key = get_encryption_key()
        encrypted = self.create_encrypted_value(plaintext, correct_key)

        # Use wrong key
        wrong_key = b"wrong_key_32_bytes_exactly_!!"
        result = decrypt_aes256(encrypted, wrong_key)
        assert result is None

    def test_decrypt_corrupted_data(self):
        """Test decryption with corrupted encrypted data."""
        # Create valid encrypted data then corrupt it
        plaintext = "secret"
        key = get_encryption_key()
        encrypted = self.create_encrypted_value(plaintext, key)

        # Corrupt the encrypted data
        corrupted = encrypted[:-5] + "XXXXX"
        result = decrypt_aes256(corrupted, key)
        assert result is None

    def test_decrypt_empty_string(self):
        """Test decryption of empty string."""
        result = decrypt_aes256("", get_encryption_key())
        assert result is None



@pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not installed")
class TestDecryptConfigField:
    """Test suite for decrypt_config_field."""

    def create_encrypted_value(self, plaintext: str, key: bytes) -> str:
        """Helper to create encrypted values for testing."""
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import padding
        import os

        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(plaintext.encode('utf-8')) + padder.finalize()
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        encrypted = base64.b64encode(iv + ciphertext).decode('utf-8')
        return encrypted

    def test_decrypt_nested_field(self):
        """Test decrypting nested config field."""
        key = get_encryption_key()
        plaintext = "my-redis-password"
        encrypted = self.create_encrypted_value(plaintext, key)

        config = {
            "redis": {
                "host": "localhost",
                "password": encrypted
            }
        }

        result = decrypt_config_field(config, "redis.password", key)
        assert result["redis"]["password"] == plaintext

    def test_decrypt_top_level_field(self):
        """Test decrypting top-level config field."""
        key = get_encryption_key()
        plaintext = "secret-value"
        encrypted = self.create_encrypted_value(plaintext, key)

        config = {"api_key": encrypted}

        result = decrypt_config_field(config, "api_key", key)
        assert result["api_key"] == plaintext

    def test_decrypt_missing_field(self):
        """Test decrypting non-existent field."""
        config = {"redis": {"host": "localhost"}}

        result = decrypt_config_field(config, "redis.password", None)
        # Should return config unchanged
        assert result == config

    def test_decrypt_missing_parent(self):
        """Test decrypting field with missing parent."""
        config = {"other": "value"}

        result = decrypt_config_field(config, "redis.password", None)
        # Should return config unchanged
        assert result == config

    def test_decrypt_non_string_value(self):
        """Test decrypting non-string value."""
        config = {"redis": {"port": 6379}}

        result = decrypt_config_field(config, "redis.port", None)
        # Should return config unchanged
        assert result["redis"]["port"] == 6379

    def test_decrypt_empty_string_value(self):
        """Test decrypting empty string value."""
        config = {"redis": {"password": ""}}

        result = decrypt_config_field(config, "redis.password", None)
        # Should return config unchanged
        assert result["redis"]["password"] == ""

    def test_decrypt_deeply_nested_field(self):
        """Test decrypting deeply nested field."""
        key = get_encryption_key()
        plaintext = "deep-secret"
        encrypted = self.create_encrypted_value(plaintext, key)

        config = {
            "level1": {
                "level2": {
                    "level3": {
                        "secret": encrypted
                    }
                }
            }
        }

        result = decrypt_config_field(config, "level1.level2.level3.secret", key)
        assert result["level1"]["level2"]["level3"]["secret"] == plaintext


@pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not installed")
class TestDecryptGlobalConfig:
    """Test suite for decrypt_global_config."""

    def create_encrypted_value(self, plaintext: str, key: bytes) -> str:
        """Helper to create encrypted values for testing."""
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import padding
        import os

        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(plaintext.encode('utf-8')) + padder.finalize()
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        encrypted = base64.b64encode(iv + ciphertext).decode('utf-8')
        return encrypted

    def test_decrypt_redis_password(self):
        """Test decrypting redis password in global config."""
        key = get_encryption_key()
        plaintext = "redis-secret-pwd"
        encrypted = self.create_encrypted_value(plaintext, key)

        config = {
            "redis": {
                "host": "localhost",
                "port": 6379,
                "password": encrypted
            }
        }

        result = decrypt_global_config(config, key)
        assert result["redis"]["password"] == plaintext
        assert result["redis"]["host"] == "localhost"
        assert result["redis"]["port"] == 6379

    def test_decrypt_no_redis_config(self):
        """Test with config that has no redis section."""
        config = {"other": "value"}

        result = decrypt_global_config(config, None)
        # Should return config unchanged
        assert result == config

    def test_decrypt_redis_without_password(self):
        """Test with redis config but no password field."""
        config = {
            "redis": {
                "host": "localhost",
                "port": 6379
            }
        }

        result = decrypt_global_config(config, None)
        # Should return config unchanged
        assert result == config

    def test_decrypt_preserves_other_fields(self):
        """Test that decryption preserves other config fields."""
        key = get_encryption_key()
        plaintext = "redis-pwd"
        encrypted = self.create_encrypted_value(plaintext, key)

        config = {
            "redis": {
                "host": "localhost",
                "password": encrypted
            },
            "database": {
                "url": "postgresql://localhost"
            },
            "api_key": "public-key"
        }

        result = decrypt_global_config(config, key)
        assert result["redis"]["password"] == plaintext
        assert result["database"]["url"] == "postgresql://localhost"
        assert result["api_key"] == "public-key"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
