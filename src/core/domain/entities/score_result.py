from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any

@dataclass
class ScoreResultEntity:
    """
    Core entity representing the screening results for a resume against a JD.
    """
    id: str
    resume_id: str
    jd_id: str
    job_id: str                   # Screening job ID
    skills_score: float           # 0-100, weight 25%
    ats_score: float              # 0-100, weight 20%
    project_score: float          # 0-100, weight 20%
    experience_score: float       # 0-100, weight 20%
    education_score: float        # 0-100, weight 15%
    final_score: float            # weighted composite 0-100
    strengths: list[str]          # 2-3 key strengths
    gaps: list[str]               # 2-3 key gaps
    matched_keywords: list[str]   # ATS keywords found
    missing_keywords: list[str]   # ATS keywords not found
    keyword_match_rate: float     # percentage
    confidence_score: float       # AI confidence 0-100
    quality_flag: str             # "high", "medium", "low"
    recommendation: str           # "Strong Fit", "Moderate Fit", "Not Fit"
    processing_time_seconds: float
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScoreResultEntity":
        """
        Creates a ScoreResultEntity from a dictionary.
        """
        return cls(
            id=data.get("id", ""),
            resume_id=data.get("resume_id", ""),
            jd_id=data.get("jd_id", ""),
            job_id=data.get("job_id", ""),
            skills_score=float(data.get("skills_score", 0.0)),
            ats_score=float(data.get("ats_score", 0.0)),
            project_score=float(data.get("project_score", 0.0)),
            experience_score=float(data.get("experience_score", 0.0)),
            education_score=float(data.get("education_score", 0.0)),
            final_score=float(data.get("final_score", 0.0)),
            strengths=data.get("strengths", []),
            gaps=data.get("gaps", []),
            matched_keywords=data.get("matched_keywords", []),
            missing_keywords=data.get("missing_keywords", []),
            keyword_match_rate=float(data.get("keyword_match_rate", 0.0)),
            confidence_score=float(data.get("confidence_score", 100.0)),
            quality_flag=data.get("quality_flag", "medium"),
            recommendation=data.get("recommendation", "Not Fit"),
            processing_time_seconds=float(data.get("processing_time_seconds", 0.0)),
            created_at=data.get("created_at", datetime.now())
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Converts ScoreResultEntity to a dictionary.
        Converts datetime objects to ISO format strings for JSON serialization.
        """
        data = asdict(self)
        if isinstance(data.get("created_at"), datetime):
            data["created_at"] = data["created_at"].isoformat()
        return data

    @property
    def is_strong_fit(self) -> bool:
        """Determines if the candidate is a strong fit."""
        return self.final_score >= 75.0

    @property
    def is_not_fit(self) -> bool:
        """Determines if the candidate is not a fit."""
        return self.final_score < 50.0
