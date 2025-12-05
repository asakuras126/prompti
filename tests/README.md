# Prompti Tests

This directory contains comprehensive unit tests for the Prompti library, organized to mirror the source code structure.

## Directory Structure

```
tests/
├── README.md                          # This file
├── __init__.py
├── conftest.py                        # Pytest configuration (if needed)
│
├── test_engine.py                     # Tests for core engine.py
├── test_replay.py                     # Tests for replay functionality
├── test_tracing.py                    # Tests for tracing
├── test_experiment.py                 # Tests for experiment
├── mock_server.py                     # Mock server utilities
│
├── loader/                            # Tests for loader module
│   ├── __init__.py
│   ├── template_loader/               # Template loader tests
│   │   ├── __init__.py
│   │   ├── test_memory_loader.py      # MemoryLoader tests
│   │   ├── test_file_loader.py        # FileLoader tests
│   │   └── test_http_loader.py        # HTTPLoader tests
│   │
│   └── model_config_loader/           # Model config loader tests
│       ├── __init__.py
│       ├── test_memory_model_config_loader.py
│       └── test_file_model_config_loader.py
│
├── model_client/                      # Tests for model clients
│   ├── __init__.py
│   └── test_config_loader.py          # Model client config tests
│
├── hooks/                             # Tests for hooks
│   ├── __init__.py
│   ├── test_safety_classification_hook.py
│   └── test_wordlist_anonymization_hook.py
│
├── router/                            # Tests for routing
│   ├── __init__.py
│   ├── test_conditional.py            # Condition tests (ListCondition, BoolCondition, etc.)
│   └── test_selector_pipeline.py      # Selector and pipeline tests
│
├── utils/                             # Tests for utilities
│   ├── __init__.py
│   └── test_encryption.py             # Encryption utility tests
│
└── template/                          # Tests for template functionality
    ├── __init__.py
    ├── test_template_format.py
    └── test_version_selector.py
```

## Test Coverage

### Core Components

#### 1. Engine (`test_engine.py`)
- Core engine functionality
- Template loading and caching
- Model configuration handling
- Direct message execution
- Variant selection

#### 2. Loader Module

**Template Loaders:**
- `test_memory_loader.py` - In-memory template storage
  - YAML format support
  - Dict format support
  - PromptTemplate instance support
  - Add/remove templates
  - Version management

- `test_file_loader.py` - File system template loading
  - YAML file parsing
  - Version matching
  - Model strategy compatibility
  - Multiple variants

- `test_http_loader.py` - HTTP-based template loading
  - Version listing
  - Template fetching
  - Error handling
  - Mock server integration

**Model Config Loaders:**
- `test_memory_model_config_loader.py` - In-memory model configs
- `test_file_model_config_loader.py` - File-based model configs
  - Single config loading
  - Model list loading
  - Model lookup by name/provider
  - Config validation

#### 3. Router Module

**Conditions (`test_conditional.py`):**
- `ListCondition` - Whitelist/blacklist matching
- `BoolCondition` - Boolean flag matching
- `ValueCondition` - Value matching (case-sensitive/insensitive)
- `MinMaxCondition` - Numeric range matching
- `HashCondition` - Consistent hash-based routing
  - Percentage-based traffic splitting
  - Fine-grained control (0.01% precision)
  - Consistent hashing

**Selectors and Pipeline (`test_selector_pipeline.py`):**
- `Candidate` - Route destination
- `WRRSelector` - Weighted Round Robin selection
  - Thread-safe selection
  - Weight distribution
- `Route` - Single routing rule
  - Condition matching
  - Quota consumption
- `RoutePipeline` - Sequential route evaluation
  - First-match routing
  - Weight/quota management
  - Logging

#### 4. Hooks Module

- `test_safety_classification_hook.py` - Safety classification
  - Sentence extraction
  - Context building
  - Streaming chunk processing
  - History buffer management

- `test_wordlist_anonymization_hook.py` - Content anonymization
  - Wordlist-based anonymization

#### 5. Utils Module

- `test_encryption.py` - Encryption utilities
  - AES-256 decryption
  - Config field decryption
  - Key management
  - Error handling

## Running Tests

### Run All Tests
```bash
cd prompti
pytest tests/
```

### Run Specific Module Tests
```bash
# Test loaders
pytest tests/loader/

# Test router
pytest tests/router/

# Test hooks
pytest tests/hooks/

# Test utils
pytest tests/utils/
```

### Run Specific Test File
```bash
pytest tests/router/test_conditional.py
```

### Run with Coverage
```bash
pytest tests/ --cov=prompti --cov-report=html
```

### Run with Verbose Output
```bash
pytest tests/ -v
```

## Test Design Principles

### 1. No External Dependencies
All tests are designed to run without external services:
- Use mocks for HTTP clients
- Use temporary directories for file operations
- Use in-memory storage where possible

### 2. Isolated Tests
Each test is independent and doesn't rely on:
- External state
- Other tests
- Network services
- Databases

### 3. Comprehensive Coverage
Tests cover:
- Happy paths (normal operation)
- Error conditions
- Edge cases
- Boundary conditions
- Thread safety (where applicable)

### 4. Clear Test Names
Test names clearly describe what is being tested:
```python
def test_condition_matches_when_value_in_allow_list(self):
    """Test that values in allow list match."""
```

### 5. Fixtures and Helpers
Use pytest fixtures for common setup:
```python
@pytest.fixture
def temp_config_file():
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "config.yaml"
```

## Key Testing Patterns

### Testing Async Functions
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result == expected
```

### Testing with Mocks
```python
from unittest.mock import Mock, patch

@patch('module.external_dependency')
def test_with_mock(mock_dependency):
    mock_dependency.return_value = "mocked"
    result = function_under_test()
    assert result == expected
```

### Testing File Operations
```python
from tempfile import TemporaryDirectory
from pathlib import Path

def test_file_operation():
    with TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.txt"
        file_path.write_text("content")
        # Test file operations
```

### Testing Exceptions
```python
def test_raises_error():
    with pytest.raises(ValueError, match="error message"):
        function_that_raises()
```

## Missing Tests (To Be Added)

The following areas need additional test coverage:

### High Priority
1. **Model Client Tests** - Test individual model client implementations:
   - `openai_client.py`
   - `qianfan_client.py`
   - `gemini_http_provider.py`
   - `litellm.py`
   - Mock provider tests
   - Factory tests

2. **Engine Advanced Features**:
   - Streaming responses
   - Hook integration
   - Error handling and retries
   - Metrics collection
   - Tool calling

### Medium Priority
3. **Template Module**:
   - Template formatting with Jinja2
   - Variable substitution
   - Variant selection logic

4. **Global Config Loader**:
   - Loading global configuration
   - Config merging

5. **Sticky Helper**:
   - Session stickiness
   - Cache management

### Low Priority
6. **Integration Tests**:
   - End-to-end workflow tests
   - Multi-component interaction tests

7. **Performance Tests**:
   - Load testing
   - Concurrent request handling

## Contributing

When adding new tests:

1. Place tests in the appropriate directory matching the source structure
2. Follow existing naming conventions
3. Add docstrings explaining what is being tested
4. Ensure tests run without external dependencies
5. Add new test files to this README
6. Maintain high test coverage (aim for >80%)

## Troubleshooting

### Import Errors
If you encounter import errors, ensure you're running from the correct directory:
```bash
cd /path/to/prompti
pytest tests/
```

### Async Test Failures
Make sure to install pytest-asyncio:
```bash
pip install pytest-asyncio
```

### Cryptography Tests
Encryption tests require the cryptography library:
```bash
pip install cryptography
```

If the library is not installed, these tests will be skipped automatically.

## Test Maintenance

- **Review tests regularly** when source code changes
- **Update mocks** when external APIs change
- **Refactor common patterns** into fixtures
- **Remove obsolete tests** when features are removed
- **Keep test data realistic** but simple
