from typing import Any, Optional
from fastapi import HTTPException

class RecruitIQException(Exception):
    """
    Base Exception for RecruitIQ application.
    """
    def __init__(self, message: str, detail: Optional[Any] = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

class OCRFailedException(RecruitIQException):
    """
    Raised when PDF text extraction (OCR) fails.
    """
    pass

class LLMTimeoutException(RecruitIQException):
    """
    Raised when the configured LLM API times out after retries.
    """
    pass


class LLMAuthenticationException(RecruitIQException):
    """Raised when configured LLM credentials are invalid or unauthorized."""
    pass

class InvalidFileTypeException(RecruitIQException):
    """
    Raised when an unsupported file type is uploaded.
    """
    pass

class FileTooLargeException(RecruitIQException):
    """
    Raised when an uploaded file exceeds the MAX_FILE_SIZE_MB limit.
    """
    pass

class ScoringException(RecruitIQException):
    """
    Raised when an error occurs during the scoring engine execution.
    """
    pass

class DatabaseException(RecruitIQException):
    """
    Raised when a database connection or query error occurs.
    """
    pass

class LLMOutputValidationException(RecruitIQException):
    """Raised when LLM output cannot be parsed/validated into expected schema."""
    pass


def map_exception_to_http(error: Exception) -> tuple[int, str, str]:
    """Centralized exception-to-HTTP mapper for API routes.

    Returns tuple: (status_code, code, message)
    """
    if isinstance(error, InvalidFileTypeException):
        return (400, "INVALID_FILE_TYPE", error.message)
    if isinstance(error, FileTooLargeException):
        return (400, "FILE_TOO_LARGE", error.message)
    if isinstance(error, OCRFailedException):
        return (422, "OCR_FAILED", error.message)
    if isinstance(error, LLMOutputValidationException):
        return (422, "LLM_OUTPUT_VALIDATION_ERROR", error.message)
    if isinstance(error, LLMTimeoutException):
        return (504, "LLM_TIMEOUT", error.message)
    if isinstance(error, LLMAuthenticationException):
        return (502, "LLM_AUTH_FAILED", error.message)
    if isinstance(error, ScoringException):
        return (422, "SCORING_ERROR", error.message)
    if isinstance(error, DatabaseException):
        return (500, "DATABASE_ERROR", error.message)
    if isinstance(error, HTTPException):
        detail = error.detail if isinstance(error.detail, str) else "HTTP error"
        return (int(error.status_code), "HTTP_ERROR", detail)
    if isinstance(error, RecruitIQException):
        return (400, type(error).__name__, error.message)
    return (500, "INTERNAL_ERROR", str(error))
