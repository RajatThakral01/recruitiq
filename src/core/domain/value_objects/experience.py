from dataclasses import dataclass, field

@dataclass(frozen=True)
class Experience:
    """
    Value object representing professional experience.
    """
    company: str
    role: str
    years: float
    description: str
    tech_stack: list[str] = field(default_factory=list)
