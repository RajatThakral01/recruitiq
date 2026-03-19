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
