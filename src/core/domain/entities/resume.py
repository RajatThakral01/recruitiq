from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Optional

@dataclass
class ResumeEntity:
    """
    Core entity representing a candidate's resume.
    """
    id: str
    candidate_name: str
    email: str
    raw_text: str
    parsed_skills: list[str]
    years_experience: float
    education_level: str  # "Bachelor", "Master", "PhD", "Other"
    projects: list[dict]  # [{name, description, tech_stack, metrics}]
    quality_flag: str     # "high", "medium", "low"
    relevant_experience: float = 0.0  # years directly relevant to the applied role
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResumeEntity":
        """
        Creates a ResumeEntity from a dictionary.
        """
        return cls(
            id=data.get("id", ""),
            candidate_name=data.get("candidate_name", ""),
            email=data.get("email", ""),
            raw_text=data.get("raw_text", ""),
            parsed_skills=data.get("parsed_skills", []),
            years_experience=float(data.get("years_experience", 0.0)),
            education_level=data.get("education_level", "Other"),
            projects=data.get("projects", []),
            quality_flag=data.get("quality_flag", "medium"),
            relevant_experience=float(data.get("relevant_experience", 0.0)),
            created_at=data.get("created_at", datetime.now())
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Converts ResumeEntity to a dictionary.
        Converts datetime objects to ISO format strings for JSON serialization.
        """
        data = asdict(self)
        if isinstance(data.get("created_at"), datetime):
            data["created_at"] = data["created_at"].isoformat()
        return data
