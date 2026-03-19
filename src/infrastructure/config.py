from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings for RecruitIQ.
    Reads values from environment variables or a .env file.
    """

    # Required
    ARCEE_API_KEY: str
    DATABASE_URL: str

    # Optional with Defaults
    LOG_LEVEL: str = Field(default="INFO")
    MAX_FILE_SIZE_MB: int = Field(default=10)
    MAX_RESUMES: int = Field(default=10)
    CORS_ORIGINS: str = Field(default="*")
    ARCEE_BASE_URL: str = Field(default="https://conductor.arcee.ai/v1")
    ARCEE_MODEL: str = Field(default="arcee-ai/Trinity-2-7B")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Singleton instance for the application
settings = Settings()
