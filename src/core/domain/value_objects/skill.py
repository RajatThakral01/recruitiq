from dataclasses import dataclass

@dataclass(frozen=True)
class Skill:
    """
    Value object representing a skill.
    """
    name: str
    level: str  # "expert", "intermediate", "beginner"
    is_required: bool
