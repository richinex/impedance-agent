# src/core/config.py
import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional


@dataclass
class APIConfig:
    api_key: str
    base_url: str
    model: str


class Environment:
    def __init__(self):
        # Try multiple locations for .env file
        possible_paths = [
            Path.cwd() / ".env",  # Current working directory
            Path(__file__).parent.parent / ".env",  # Package directory
            Path(__file__).parent.parent.parent / ".env",  # Project root
        ]

        # Use the first .env file found
        env_path = next((path for path in possible_paths if path.exists()), None)

        if env_path:
            load_dotenv(env_path)
        else:
            # Optional: print a warning or hint about .env.example
            example_path = Path(__file__).parent.parent / ".env.example"
            if example_path.exists():
                print(
                    f"No .env file found. Please copy {example_path} to create your .env file."
                )

        # API Configurations
        self.deepseek = APIConfig(
            api_key=self._get_optional("DEEPSEEK_API_KEY", ""),
            base_url=self._get_optional(
                "DEEPSEEK_API_BASE_URL", "https://api.deepseek.com"
            ),
            model="deepseek-chat",
        )

        self.openai = APIConfig(
            api_key=self._get_optional("OPENAI_API_KEY", ""),
            base_url=self._get_optional(
                "OPENAI_API_BASE_URL", "https://api.openai.com/v1"
            ),
            model="gpt-4o-mini",
        )

        # Other configurations
        self.log_level = self._get_optional("LOG_LEVEL", "INFO")

        # Validate configurations
        self._validate_configs()

    def _get_required(self, key: str) -> str:
        """
        Get a required environment variable.

        Args:
            key: The environment variable key

        Returns:
            The environment variable value

        Raises:
            ValueError: If the environment variable is not set
        """
        value = os.getenv(key)
        if value is None:
            raise ValueError(
                f"Required environment variable {key} is not set. "
                "Please make sure you have copied .env.example to .env "
                "and set your API keys."
            )
        return value

    def _get_optional(self, key: str, default: str) -> str:
        """
        Get an optional environment variable with a default value.

        Args:
            key: The environment variable key
            default: The default value if not set

        Returns:
            The environment variable value or default
        """
        return os.getenv(key, default)

    def _validate_configs(self) -> None:
        """
        Validate the configuration settings.

        This ensures that at least one provider has valid credentials.
        """
        has_deepseek = bool(self.deepseek.api_key.strip())
        has_openai = bool(self.openai.api_key.strip())

        if not (has_deepseek or has_openai):
            raise ValueError(
                "No valid API credentials found. "
                "Please set either DEEPSEEK_API_KEY or OPENAI_API_KEY in your .env file."
            )

    def get_provider_config(self, provider: str) -> Optional[APIConfig]:
        """
        Get the configuration for a specific provider.

        Args:
            provider: The name of the provider ('deepseek' or 'openai')

        Returns:
            The APIConfig for the specified provider or None if not found

        Raises:
            ValueError: If the provider is not supported
        """
        if provider == "deepseek":
            return self.deepseek
        elif provider == "openai":
            return self.openai
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def is_provider_configured(self, provider: str) -> bool:
        """
        Check if a provider is properly configured.

        Args:
            provider: The name of the provider to check

        Returns:
            True if the provider has valid credentials, False otherwise
        """
        try:
            config = self.get_provider_config(provider)
            return bool(config and config.api_key.strip())
        except ValueError:
            return False

    def get_available_providers(self) -> list[str]:
        """
        Get a list of configured providers.

        Returns:
            List of provider names that have valid credentials
        """
        providers = []
        if self.is_provider_configured("deepseek"):
            providers.append("deepseek")
        if self.is_provider_configured("openai"):
            providers.append("openai")
        return providers


# Global environment instance
env = Environment()
