from typing import Optional
from pydantic import BaseModel


class ScreenRequest(BaseModel):
    """Request body for POST /api/screen."""
    jd_id: str
    resume_ids: list[str]


class CandidateResult(BaseModel):
    """Represents one candidate's ranked screening result."""
    rank: int
    resume_id: str
    candidate_name: str
    email: str
    final_score: float
    skills_score: float
    ats_score: float
    project_score: float
    experience_score: float
    education_score: float
    strengths: list[str]
    gaps: list[str]
    matched_keywords: list[str]
    missing_keywords: list[str]
    keyword_match_rate: float
    confidence_score: float
    quality_flag: str
    recommendation: str
    processing_time_seconds: float


class ScreeningResponse(BaseModel):
    """Response returned by the /api/screen and /api/results endpoints."""
    job_id: str
    status: str
    jd_title: str
    total_screened: int
    results: list[CandidateResult]


class UploadJDResponse(BaseModel):
    """Response returned after a successful JD upload."""
    jd_id: str
    title: str
    keywords_extracted: int
    required_skills: list[str]


class UploadResumesResponse(BaseModel):
    """Response returned after successful resume uploads."""
    resume_ids: list[str]
    files_processed: int


class ErrorResponse(BaseModel):
    """Consistent error response shape for all error cases."""
    error: bool = True
    message: str
    code: str
