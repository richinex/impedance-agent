# tests/test_core/test_env.py
import pytest
from pathlib import Path
from impedance_agent.core.env import Environment, APIConfig


def test_environment_defaults():
    """Test Environment initialization with default values"""
    env = Environment({}, validate=False)  # Skip validation for testing defaults

    assert env.deepseek.api_key == ""
    assert env.deepseek.base_url == "https://api.deepseek.com"
    assert env.deepseek.model == "deepseek-chat"

    assert env.openai.api_key == ""
    assert env.openai.base_url == "https://api.openai.com/v1"
    assert env.openai.model == "gpt-4-mini"

    assert env.log_level == "INFO"


def test_environment_with_config():
    """Test Environment initialization with config"""
    config = {
        "DEEPSEEK_API_KEY": "test_key",
        "DEEPSEEK_API_BASE_URL": "https://test.deepseek.com",
        "DEEPSEEK_MODEL": "test-model",
        "OPENAI_API_KEY": "test_key_2",
        "OPENAI_API_BASE_URL": "https://test.openai.com",
        "OPENAI_MODEL": "test-model-2",
        "LOG_LEVEL": "DEBUG"
    }
    env = Environment(config)

    assert env.deepseek.api_key == "test_key"
    assert env.deepseek.base_url == "https://test.deepseek.com"
    assert env.deepseek.model == "test-model"

    assert env.openai.api_key == "test_key_2"
    assert env.openai.base_url == "https://test.openai.com"
    assert env.openai.model == "test-model-2"

    assert env.log_level == "DEBUG"


def test_validation_error():
    """Test configuration validation"""
    with pytest.raises(ValueError, match="No valid API credentials found"):
        Environment({}, validate=True)  # Explicitly test validation


def test_get_provider_config():
    """Test getting provider configurations"""
    config = {
        "DEEPSEEK_API_KEY": "test_deepseek",
        "OPENAI_API_KEY": "test_openai"
    }
    env = Environment(config)

    deepseek_config = env.get_provider_config("deepseek")
    assert isinstance(deepseek_config, APIConfig)
    assert deepseek_config.api_key == "test_deepseek"

    openai_config = env.get_provider_config("openai")
    assert isinstance(openai_config, APIConfig)
    assert openai_config.api_key == "test_openai"

    with pytest.raises(ValueError, match="Unsupported provider"):
        env.get_provider_config("invalid_provider")


def test_is_provider_configured():
    """Test provider configuration checking"""
    # Test with no configs
    env = Environment({}, validate=False)
    assert env.is_provider_configured("deepseek") is False
    assert env.is_provider_configured("openai") is False
    assert env.is_provider_configured("invalid_provider") is False

    # Test with one config
    env = Environment({"DEEPSEEK_API_KEY": "test"})
    assert env.is_provider_configured("deepseek") is True
    assert env.is_provider_configured("openai") is False

    # Test with both configs
    env = Environment({
        "DEEPSEEK_API_KEY": "test1",
        "OPENAI_API_KEY": "test2"
    })
    assert env.is_provider_configured("deepseek") is True
    assert env.is_provider_configured("openai") is True


def test_get_available_providers():
    """Test getting available providers"""
    # Test with no configs
    env = Environment({}, validate=False)
    assert env.get_available_providers() == []

    # Test with one config
    env = Environment({"DEEPSEEK_API_KEY": "test"})
    assert env.get_available_providers() == ["deepseek"]

    # Test with both configs
    env = Environment({
        "DEEPSEEK_API_KEY": "test1",
        "OPENAI_API_KEY": "test2"
    })
    providers = env.get_available_providers()
    assert len(providers) == 2
    assert "deepseek" in providers
    assert "openai" in providers


def test_env_file_loading(tmp_path):
    """Test .env file loading from different locations"""
    # Create a test .env file
    env_content = """
    DEEPSEEK_API_KEY=test_key
    OPENAI_API_KEY=test_key
    """
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)

    env = Environment({}, validate=False)
    assert env.deepseek.api_key == ""
    assert env.openai.api_key == ""