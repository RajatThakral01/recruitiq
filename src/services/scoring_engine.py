import uuid
import re
from datetime import datetime
from src.core.domain.entities.resume import ResumeEntity
from src.core.domain.entities.job_description import JDEntity
from src.core.domain.entities.score_result import ScoreResultEntity
from src.services.ats_scorer import ATSScorer
from src.services.semantic_scorer import SemanticScorer
from src.infrastructure.logger import logger
from src.infrastructure.exceptions import ScoringException


EDUCATION_RANK = {
    "PhD": 4,
    "Master": 3,
    "Bachelor": 2,
    "Other": 1,
}

SECTION_HEADERS = ["experience", "education", "skills", "summary", "projects", "work history"]
WEIGHTS = {
    "skills": 0.25,
    "ats": 0.20,
    "projects": 0.20,
    "experience": 0.20,
    "education": 0.15,
}


class ScoringEngine:
    """
    Orchestrates all scoring dimensions and compiles a complete ScoreResultEntity.
    """

    def __init__(self, ats_scorer: ATSScorer, semantic_scorer: SemanticScorer) -> None:
        """Injects the ATS and semantic scorer dependencies."""
        self.ats_scorer = ats_scorer
        self.semantic_scorer = semantic_scorer

    # ────────────────────────────────────────────────────────────
    # Individual dimension scorers
    # ────────────────────────────────────────────────────────────

    def compute_experience_score(self, resume: ResumeEntity, jd: JDEntity) -> float:
        """
        Grades candidate experience vs the JD's minimum experience requirement.

        Returns a float score 0-100.
        """
        years = resume.years_experience
        required = jd.min_experience

        if required <= 0:
            logger.info("JD has no minimum experience requirement — awarding full experience score.")
            return 100.0

        if years >= required * 1.5:
            score = 100.0
        elif years >= required:
            score = 85.0
        elif years >= required * 0.7:
            score = 60.0
        elif years >= required * 0.5:
            score = 40.0
        else:
            score = 20.0

        logger.info(f"Experience score: {score} ({years} yrs vs {required} required)")
        return score

    def compute_education_score(self, resume: ResumeEntity, jd: JDEntity) -> float:
        """
        Grades candidate education level vs the JD's education requirement.

        Returns a float score 0-100.
        """
        resume_rank = EDUCATION_RANK.get(resume.education_level, 1)
        required_rank = EDUCATION_RANK.get(jd.education_requirement, 1)

        diff = required_rank - resume_rank

        if diff <= 0:
            score = 100.0
        elif diff == 1:
            score = 70.0
        elif diff == 2:
            score = 40.0
        else:
            score = 20.0

        logger.info(
            f"Education score: {score} "
            f"(resume='{resume.education_level}' rank={resume_rank}, "
            f"required='{jd.education_requirement}' rank={required_rank})"
        )
        return score

    def compute_final_score(
        self,
        skills_score: float,
        ats_score: float,
        project_score: float,
        experience_score: float,
        education_score: float,
    ) -> float:
        """
        Computes weighted composite score from the 5 dimensions.

        Returns a float 0-100.
        """
        final = (
            skills_score * WEIGHTS["skills"]
            + ats_score * WEIGHTS["ats"]
            + project_score * WEIGHTS["projects"]
            + experience_score * WEIGHTS["experience"]
            + education_score * WEIGHTS["education"]
        )
        return round(final, 2)

    def get_recommendation(self, final_score: float) -> str:
        """
        Returns a hire recommendation string based on final score.
        """
        if final_score >= 75:
            return "Strong Fit"
        elif final_score >= 50:
            return "Moderate Fit"
        return "Not Fit"

    def get_quality_flag(self, resume_text: str) -> str:
        """
        Determines resume quality based on word count and section coverage.

        Returns "high", "medium", or "low".
        """
        word_count = len(resume_text.split())
        lower_text = resume_text.lower()
        sections_found = sum(1 for h in SECTION_HEADERS if h in lower_text)

        if word_count >= 400 and sections_found >= 3:
            return "high"
        elif word_count >= 200:
            return "medium"
        return "low"

    # ────────────────────────────────────────────────────────────
    # Orchestration
    # ────────────────────────────────────────────────────────────

    def compute_all(
        self,
        resume: ResumeEntity,
        jd: JDEntity,
        resume_text: str,
    ) -> ScoreResultEntity:
        """
        Runs all scoring dimensions and assembles a complete ScoreResultEntity.

        Steps:
          1. ATS keyword scoring
          2. Semantic skills scoring
          3. Project quality scoring
          4. Experience scoring
          5. Education scoring
          6. Composite final score
          7. Recommendation + quality flag
          8. AI confidence
        """
        try:
            logger.info(
                f"Starting full scoring pipeline for resume_id={resume.id}, jd_id={jd.id}"
            )

            # 1. ATS
            logger.info("Step 1/8 — ATS scoring")
            ats_result = self.ats_scorer.score(resume_text, jd)
            ats_score = ats_result["ats_score"]
            matched_keywords = ats_result["matched_keywords"]
            missing_keywords = ats_result["missing_keywords"]
            keyword_match_rate = ats_result["keyword_match_rate"]

            # 2. Semantic skills
            logger.info("Step 2/8 — Semantic skills scoring")
            skills_score = self.semantic_scorer.score_skills(resume_text, jd)

            # 3. Project quality
            logger.info("Step 3/8 — Project quality scoring")
            project_score = self.semantic_scorer.score_projects(resume_text, jd)

            # 4. Experience
            logger.info("Step 4/8 — Experience scoring")
            experience_score = self.compute_experience_score(resume, jd)

            # 5. Education
            logger.info("Step 5/8 — Education scoring")
            education_score = self.compute_education_score(resume, jd)

            # 6. Composite score
            logger.info("Step 6/8 — Computing final composite score")
            final_score = self.compute_final_score(
                skills_score, ats_score, project_score, experience_score, education_score
            )

            # 7. Recommendation + quality flag
            logger.info("Step 7/8 — Computing recommendation and quality flag")
            recommendation = self.get_recommendation(final_score)
            quality_flag = self.get_quality_flag(resume_text)

            # 8. Confidence
            logger.info("Step 8/8 — Computing AI confidence")
            all_scores = {
                "skills": skills_score,
                "ats": ats_score,
                "projects": project_score,
                "experience": experience_score,
                "education": education_score,
            }
            confidence_score = self.semantic_scorer.get_confidence(all_scores)

            logger.info(
                f"Scoring complete — final={final_score}, "
                f"recommendation={recommendation}, quality={quality_flag}"
            )

            return ScoreResultEntity(
                id=str(uuid.uuid4()),
                resume_id=resume.id,
                jd_id=jd.id,
                job_id="",             # To be populated by use case
                skills_score=skills_score,
                ats_score=ats_score,
                project_score=project_score,
                experience_score=experience_score,
                education_score=education_score,
                final_score=final_score,
                strengths=[],           # populated by LLM adapter post-scoring
                gaps=[],                # populated by LLM adapter post-scoring
                matched_keywords=matched_keywords,
                missing_keywords=missing_keywords,
                keyword_match_rate=keyword_match_rate,
                confidence_score=confidence_score,
                quality_flag=quality_flag,
                recommendation=recommendation,
                processing_time_seconds=0.0,  # set by the calling use case
                created_at=datetime.now(),
            )

        except ScoringException:
            raise
        except Exception as e:
            logger.error(f"Scoring pipeline failed: {str(e)}", exc_info=True)
            raise ScoringException("Full scoring pipeline failed.", detail=str(e))
