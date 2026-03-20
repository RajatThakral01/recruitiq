from abc import ABC, abstractmethod
from src.core.domain.entities.resume import ResumeEntity
from src.core.domain.entities.job_description import JDEntity
from src.core.domain.entities.score_result import ScoreResultEntity

class StoragePort(ABC):
    """
    Abstract port for data persistence (PostgreSQL).
    """

    @abstractmethod
    def save_resume(self, resume: ResumeEntity) -> str:
        """Saves a parsed resume to the database."""
        pass

    @abstractmethod
    def save_jd(self, jd: JDEntity) -> str:
        """Saves a parsed job description to the database."""
        pass

    @abstractmethod
    def get_jd(self, jd_id: str) -> JDEntity:
        """Retrieves a single job description by its ID."""
        pass

    @abstractmethod
    def save_score(self, score: ScoreResultEntity) -> str:
        """Saves a screening result to the database."""
        pass

    @abstractmethod
    def get_scores_by_job(self, job_id: str) -> list[ScoreResultEntity]:
        """Retrieves all screening results for a specific job."""
        pass

    @abstractmethod
    def get_resume(self, resume_id: str) -> ResumeEntity:
        """Retrieves a single resume by its ID."""
        pass

    @abstractmethod
    def create_screening_job(self, jd_id: str) -> str:
        """Registers a new asynchronous screening job."""
        pass

    @abstractmethod
    def update_job_status(self, job_id: str, status: str) -> None:
        """Updates the status of an ongoing screening job."""
        pass

    @abstractmethod
    def job_exists(self, job_id: str) -> bool:
        """Returns True if a screening job with the given ID exists."""
        pass
