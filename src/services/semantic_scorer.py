import re
import statistics
from rapidfuzz import fuzz
from src.core.domain.entities.job_description import JDEntity
from src.core.ports.embedding_port import EmbeddingPort
from src.infrastructure.logger import logger
from src.infrastructure.exceptions import ScoringException


IMPACT_WORDS = [
    "built", "scaled", "improved", "reduced", "increased",
    "led", "designed", "developed", "launched", "optimized", "automated"
]

METRIC_PATTERN = re.compile(
    r"(\d+%|\d+\s?ms|\d+\s?users?|\$\s?\d+|\d+x\s?faster)",
    re.IGNORECASE
)


class SemanticScorer:
    """
    Scores resume sections (skills, projects) using sentence-transformers embeddings
    and heuristic project quality signals.
    """

    def __init__(self, embedding_port: EmbeddingPort) -> None:
        """
        Args:
            embedding_port: Port used for generating embeddings and calculating similarity.
        """
        self.embedding_port = embedding_port
        logger.info("SemanticScorer initialized with EmbeddingPort.")

    # ────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────

    def _extract_section(self, text: str, *headers: str, fallback_chars: int = 500) -> str:
        """
        Extracts the text that follows the first matching section header.
        Falls back to the first `fallback_chars` characters, or the full text.
        """
        lower_text = text.lower()
        for header in headers:
            idx = lower_text.find(header)
            if idx != -1:
                section_text = text[idx + len(header):]
                # Trim at the next header-like line (all-caps or short header word)
                next_section = re.search(
                    r"\n[A-Z][A-Z\s]{3,}\n|\n(?:experience|education|projects|summary|skills|"
                    r"certifications|awards|references)\b",
                    section_text, re.IGNORECASE
                )
                if next_section:
                    section_text = section_text[: next_section.start()]
                return section_text.strip()
        return text[:fallback_chars]

    # ────────────────────────────────────────────────────────────
    # Public scoring methods
    # ────────────────────────────────────────────────────────────

    def score_skills(self, resume_text: str, jd: JDEntity) -> float:
        """
        Computes semantic similarity between the JD's required skills and
        the resume's skills section.

        Returns a float score 0-100.
        """
        try:
            logger.info(f"Semantic skills scoring for JD: {jd.id}")

            jd_skills_text = " ".join(jd.required_skills)
            resume_skills_text = self._extract_section(resume_text, "skills", fallback_chars=500)

            if not jd_skills_text.strip() or not resume_skills_text.strip():
                return 0.0

            similarity = self.embedding_port.get_similarity(jd_skills_text, resume_skills_text)
            
            # Arcee Adapter returns float 0-1 (already clamped)
            # scale to 0-100
            score = round(similarity * 100, 2)

            logger.info(f"Semantic skills score: {score}")
            return score

        except Exception as e:
            logger.error(f"Semantic skills scoring failed: {str(e)}", exc_info=True)
            raise ScoringException("Semantic skills scoring failed.", detail=str(e))

    def score_projects(self, resume_text: str, jd: JDEntity) -> float:
        """
        Scores resume project/experience quality based on:
          - Impact word presence (max 40 pts)
          - Quantitative metrics (max 30 pts)
          - Tech stack relevance to JD (max 30 pts)

        Returns a float score 0-100.
        """
        try:
            logger.info(f"Project quality scoring for JD: {jd.id}")

            # Extract relevant text
            project_text = self._extract_section(
                resume_text, "projects", "experience", "work history", fallback_chars=1000
            ).lower()

            # Impact words
            impact_words_found = sum(1 for word in IMPACT_WORDS if word in project_text)
            impact_score = min(40.0, (impact_words_found / 3) * 40)

            # Metrics
            metrics_found = len(METRIC_PATTERN.findall(resume_text))
            metric_score = min(30.0, metrics_found * 10)

            # Tech relevance (fuzzy match JD required skills in project text)
            if jd.required_skills:
                matching_skills = sum(
                    1
                    for skill in jd.required_skills
                    if fuzz.partial_ratio(skill.lower(), project_text) >= 75
                )
                tech_score = min(30.0, (matching_skills / len(jd.required_skills)) * 30)
            else:
                tech_score = 0.0

            total = round(impact_score + metric_score + tech_score, 2)

            logger.info(
                f"Project score: {total} "
                f"(impact={impact_score}, metric={metric_score}, tech={tech_score})"
            )
            return total

        except Exception as e:
            logger.error(f"Project scoring failed: {str(e)}", exc_info=True)
            raise ScoringException("Project scoring failed.", detail=str(e))

    def get_confidence(self, all_scores: dict) -> float:
        """
        Computes AI confidence from the spread of dimension scores.
        Lower confidence when scores are highly uneven.

        Returns a float 0-100.
        """
        try:
            values = list(all_scores.values())
            if len(values) < 2:
                return 100.0
            std_dev = statistics.stdev(values)
            confidence = round(max(60.0, 100.0 - std_dev), 2)
            logger.info(f"Confidence score: {confidence} (std_dev={round(std_dev, 2)})")
            return confidence
        except Exception as e:
            logger.error(f"Confidence calculation failed: {str(e)}", exc_info=True)
            raise ScoringException("Confidence calculation failed.", detail=str(e))
