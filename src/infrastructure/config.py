from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator


class Settings(BaseSettings):
    """Application settings for RecruitIQ.
    Reads values from environment variables or a .env file.
    """

    # Required
    DATABASE_URL: str

    # LLM provider selection
    LLM_PROVIDER: str = Field(default="mistral")
    SCORING_MODE: str = Field(default="hybrid")

    # Optional with Defaults
    LOG_LEVEL: str = Field(default="INFO")
    MAX_FILE_SIZE_MB: int = Field(default=10)
    MAX_RESUMES: int = Field(default=10)
    CORS_ORIGINS: str = Field(default="*")
    LLM_TIMEOUT_SECONDS: float = Field(default=45.0)
    LLM_MAX_RETRIES: int = Field(default=3)
    LLM_BACKOFF_BASE_SECONDS: float = Field(default=1.5)
    MISTRAL_API_KEY: str = Field(default="")
    MISTRAL_BASE_URL: str = Field(default="https://api.mistral.ai/v1")
    MISTRAL_MODEL: str = Field(default="mistral-large-latest")
    GROK_API_KEY: str = Field(default="")
    GROK_BASE_URL: str = Field(default="https://api.x.ai/v1")
    GROK_MODEL: str = Field(default="grok-2-latest")
    GROQ_API_KEY: str = Field(default="")
    GROQ_BASE_URL: str = Field(default="https://api.groq.com/openai/v1")
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile")

    @field_validator("SCORING_MODE", mode="before")
    @classmethod
    def normalize_scoring_mode(cls, value: str) -> str:
        mode = str(value or "legacy").strip().lower()
        if mode in {"legacy", "hybrid", "llm_first"}:
            return mode
        return "legacy"

    @model_validator(mode="after")
    def validate_required_runtime_configuration(self):
        if not str(self.DATABASE_URL or "").strip():
            raise ValueError("DATABASE_URL must be set and non-empty.")

        provider = str(self.LLM_PROVIDER or "mistral").strip().lower()
        mistral_key = str(self.MISTRAL_API_KEY or "").strip()
        groq_key = str(self.GROQ_API_KEY or "").strip()
        grok_key = str(self.GROK_API_KEY or "").strip()

        if provider == "mistral" and not (mistral_key or groq_key or grok_key):
            raise ValueError(
                "MISTRAL_API_KEY (fallback: GROQ_API_KEY or GROK_API_KEY) must be set when LLM_PROVIDER=mistral."
            )
        if provider == "groq" and not (groq_key or mistral_key or grok_key):
            raise ValueError(
                "GROQ_API_KEY (fallback: MISTRAL_API_KEY or GROK_API_KEY) must be set when LLM_PROVIDER=groq."
            )
        if provider == "grok" and not (grok_key or mistral_key or groq_key):
            raise ValueError(
                "GROK_API_KEY (fallback: MISTRAL_API_KEY or GROQ_API_KEY) must be set when LLM_PROVIDER=grok."
            )
        if provider not in {"mistral", "grok", "groq"}:
            raise ValueError("LLM_PROVIDER must be one of: mistral, grok, groq.")

        return self

    model_config = SettingsConfigDict(
        env_file=(".env.example", ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Singleton instance for the application
settings = Settings()
