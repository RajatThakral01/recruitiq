import uuid
from datetime import datetime

from src.core.domain.entities.job_description import JDEntity
from src.core.ports.llm_port import LLMPort
from src.core.ports.storage_port import StoragePort
from src.infrastructure.logger import logger


class ParseJDUseCase:
    """
    Orchestrates the parsing of a raw job description text into a JDEntity.
    """

    def __init__(self, llm_port: LLMPort, storage_port: StoragePort) -> None:
        self.llm_port = llm_port
        self.storage_port = storage_port

    def execute(self, text: str) -> JDEntity:
        """
        Parse raw JD text, persist the entity, and return it.

        Args:
            text: Raw text of the job description.

        Returns:
            JDEntity populated with parsed data.
        """
        logger.info("ParseJDUseCase: parsing job description text.")
        parsed = self.llm_port.parse_jd(text)

        jd = JDEntity(
            id=str(uuid.uuid4()),
            title=parsed.get("title", ""),
            required_skills=parsed.get("required_skills", []),
            preferred_skills=parsed.get("preferred_skills", []),
            min_experience=float(parsed.get("min_experience", 0.0)),
            education_requirement=parsed.get("education_requirement", "Other"),
            keywords=parsed.get("keywords", []),
            must_have=parsed.get("must_have", []),
            nice_to_have=parsed.get("nice_to_have", []),
            seniority_level=parsed.get("seniority_level", "junior"),
            raw_text=text,
            created_at=datetime.now(),
        )

        self.storage_port.save_jd(jd)
        logger.info(f"JD parsed successfully: {jd.title}")
        return jd
