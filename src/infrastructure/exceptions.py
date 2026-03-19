from typing import Any, Optional

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
    Raised when the Arcee AI API times out after retries.
    """
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
