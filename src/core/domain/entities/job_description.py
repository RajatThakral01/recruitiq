from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any

@dataclass
class JDEntity:
    """
    Core entity representing a Job Description (JD).
    """
    id: str
    title: str
    required_skills: list[str]
    preferred_skills: list[str]
    min_experience: float
    education_requirement: str
    keywords: list[str]           # All important terms for ATS
    must_have: list[str]          # Non-negotiable requirements
    nice_to_have: list[str]       # Preferred but not required
    seniority_level: str          # "junior", "mid", "senior", "lead"
    raw_text: str                 # Full job description text
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JDEntity":
        """
        Creates a JDEntity from a dictionary.
        """
        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            required_skills=data.get("required_skills", []),
            preferred_skills=data.get("preferred_skills", []),
            min_experience=float(data.get("min_experience", 0.0)),
            education_requirement=data.get("education_requirement", "Other"),
            keywords=data.get("keywords", []),
            must_have=data.get("must_have", []),
            nice_to_have=data.get("nice_to_have", []),
            seniority_level=data.get("seniority_level", "junior"),
            raw_text=data.get("raw_text", ""),
            created_at=data.get("created_at", datetime.now())
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Converts JDEntity to a dictionary.
        Converts datetime objects to ISO format strings for JSON serialization.
        """
        data = asdict(self)
        if isinstance(data.get("created_at"), datetime):
            data["created_at"] = data["created_at"].isoformat()
        return data
