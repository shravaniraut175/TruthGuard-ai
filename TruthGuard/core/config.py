# config.py - Configuration management using pydantic-settings

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    openrouter_api_key: str = Field(default="", env="OPENROUTER_API_KEY")
    google_api_key: str = Field(default="", env="GOOGLE_API_KEY")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    
    # Base model configuration (for generating responses)
    base_provider: str = Field(default="openrouter", env="BASE_PROVIDER")
    base_model: str = Field(default="qwen/qwen2.5-72b-instruct", env="BASE_MODEL")
    
    # Judge model configuration (for evaluating responses)
    judge_provider: str = Field(default="openrouter", env="JUDGE_PROVIDER")
    judge_model: str = Field(default="anthropic/claude-3.5-sonnet", env="JUDGE_MODEL")
    
    # Search configuration
    search_provider: str = Field(default="duckduckgo", env="SEARCH_PROVIDER")
    max_search_results: int = Field(default=5, env="MAX_SEARCH_RESULTS")
    num_blackbox_samples: int = Field(default=3, env="NUM_BLACKBOX_SAMPLES")
    
    # White-box model configuration
    whitebox_model: str = Field(default="Qwen/Qwen2.5-7B-Instruct", env="WHITEBOX_MODEL")
    
    # Hallucination threshold for regeneration
    hallucination_threshold: float = Field(default=0.60, env="HALLUCINATION_THRESHOLD")
    
    # Score fusion weights
    blackbox_weight: float = Field(default=0.20, env="BLACKBOX_WEIGHT")
    whitebox_weight: float = Field(default=0.10, env="WHITEBOX_WEIGHT")
    judge_weight: float = Field(default=0.35, env="JUDGE_WEIGHT")
    grounding_weight: float = Field(default=0.35, env="GROUNDING_WEIGHT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def validate_api_keys(self) -> dict[str, str]:
        """Validate that required API keys are present."""
        missing = []
        
        if self.base_provider == "openrouter" and not self.openrouter_api_key:
            missing.append("OPENROUTER_API_KEY")
        elif self.base_provider == "google" and not self.google_api_key:
            missing.append("GOOGLE_API_KEY")
        elif self.base_provider == "openai" and not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        
        if self.judge_provider == "openrouter" and not self.openrouter_api_key:
            if "OPENROUTER_API_KEY" not in missing:
                missing.append("OPENROUTER_API_KEY")
        elif self.judge_provider == "google" and not self.google_api_key:
            if "GOOGLE_API_KEY" not in missing:
                missing.append("GOOGLE_API_KEY")
        elif self.judge_provider == "openai" and not self.openai_api_key:
            if "OPENAI_API_KEY" not in missing:
                missing.append("OPENAI_API_KEY")
        
        if missing:
            return {"valid": False, "missing_keys": missing}
        return {"valid": True, "missing_keys": []}
    
    def get_normalized_weights(self, whitebox_available: bool = True) -> dict[str, float]:
        """Get normalized weights for score fusion."""
        weights = {
            "blackbox": self.blackbox_weight,
            "whitebox": self.whitebox_weight if whitebox_available else 0.0,
            "judge": self.judge_weight,
            "grounding": self.grounding_weight
        }
        
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
