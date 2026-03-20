from abc import ABC, abstractmethod

class LLMPort(ABC):
    """
    Abstract port for Large Language Model (LLM) operations.
    """
    @abstractmethod
    def parse_resume(self, text: str) -> dict:
        """
        Parses raw resume text into a structured dictionary.
        """
        pass

    @abstractmethod
    def parse_jd(self, text: str) -> dict:
        """
        Parses raw job description text into a structured dictionary.
        """
        pass

    @abstractmethod
    def generate_strengths_gaps(self, resume: dict, jd: dict, scores: dict) -> dict:
        """
        Generates contextual strengths and gaps based on candidates match.
        """
        pass

    @abstractmethod
    def evaluate_skills_match(self, resume_text: str, jd: dict) -> dict:
        """LLM evaluation of skills match and technology transferability."""
        pass

    @abstractmethod
    def evaluate_project_quality(self, resume_text: str, jd: dict) -> dict:
        """LLM evaluation of project depth, impact and relevance."""
        pass

    @abstractmethod
    def evaluate_experience_relevance(self, resume_text: str, jd: dict) -> dict:
        """LLM evaluation of experience relevance, career trajectory and domain match."""
        pass

    @abstractmethod
    def evaluate_all_parallel(self, resume_text: str, jd: dict) -> dict:
        """Run all 3 evaluations and return combined results dict with keys:
        skills_score, project_score, experience_score"""
        pass
