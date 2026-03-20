import uuid
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
    "skills": 0.30,
    "ats": 0.15,
    "projects": 0.20,
    "experience": 0.25,
    "education": 0.10,
}

# ── Experience scorer signal lists ────────────────────────────────────────
FULLTIME_SIGNALS = [
    'full-time', 'full time', 'permanent',
    'software engineer', 'software developer',
    'data engineer', 'product manager',
    'sde', 'sde i', 'sde ii', 'sde iii',
    'engineer i', 'engineer ii',
    'senior engineer', 'staff engineer',
    'principal engineer', 'tech lead',
    'engineering lead', 'team lead',
]
INTERN_SIGNALS = [
    'intern', 'internship', 'trainee',
    'apprentice', 'co-op', 'coop',
    'summer intern', 'winter intern',
    'research intern', 'student',
]
PROGRESSION_SIGNALS = [
    # Movement and growth signals only.
    # Terms removed for being duplicates of ats_scorer.py SENIOR_SIGNALS:
    # 'lead', 'senior', 'principal', 'staff', 'mentored' (previous pass)
    # 'founded', 'head of', 'director', 'led team' (this pass)
    'promoted', 'promotion',
    'grew', 'growth', 'expanded', 'progressed',
    'increased responsibility', 'took ownership',
    'co-founded', 'started',
    'managed team',
    'team of', 'reports to', 'reported to',
]
COMPANY_TIER_SIGNALS = [
    # Global tech / finance
    'google', 'microsoft', 'amazon', 'meta',
    'apple', 'netflix', 'uber', 'airbnb',
    'walmart', 'goldman sachs', 'jp morgan', 'morgan stanley',
    'accenture', 'deloitte', 'mckinsey', 'bcg',
    # Indian unicorns / well-known startups
    'flipkart', 'swiggy', 'zomato', 'paytm',
    'razorpay', 'cred', 'zepto', 'meesho',
    'phonepe', 'groww', 'freshworks', 'zoho',
    'byju', 'ola', 'nykaa', 'myntra',
    'snapdeal', 'makemytrip', 'policybazaar', 'delhivery',
    'browserstack', 'postman', 'chargebee', 'hasura',
    'setu', 'niyo', 'slice', 'smallcase',
    # Indian IT services
    'infosys', 'tcs', 'wipro', 'hcl',
    # Institutions
    'iit', 'iim', 'nit', 'bits pilani',
    # Funding stage (meaningful signals)
    'series a', 'series b', 'series c',
]


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

    def compute_experience_score(
        self,
        resume: ResumeEntity,
        jd: JDEntity,
        resume_text: str = "",
    ) -> float:
        """
        Multi-signal experience quality scorer.

			f"skills:{skills_score}×0.30={skills_score*0.30:.1f} "
			f"ats:{ats_score}×0.15={ats_score*0.15:.1f} "
			f"proj:{project_score}×0.20={project_score*0.20:.1f} "
			f"exp:{experience_score}×0.25={experience_score*0.25:.1f} "
			f"edu:{education_score}×0.10={education_score*0.10:.1f} "
        Total                                    up to 100 pts

        Returns a float score 0-100.
        """
        lower_text = resume_text.lower()

        # ── SIGNAL 1: Years of relevant experience (max 35 pts) ───────────
        relevant = getattr(resume, 'relevant_experience', None)
        if relevant is not None and relevant > 0:
            years = relevant
            years_source = "relevant"
        elif relevant == 0.0:
            # LLM explicitly returned 0.0 → different domain; do NOT fall back
            years = 0.0
            years_source = "relevant=0 (different domain)"
        else:
            years = resume.years_experience
            years_source = "total (no relevant data)"
        required = jd.min_experience

        if required <= 0:
            years_score = 35  # no minimum set — full marks
        elif years >= required * 1.5:
            years_score = 35
        elif years >= required:
            years_score = 28
        elif years >= required * 0.7:
            years_score = 18
        elif years >= required * 0.5:
            years_score = 10
        else:
            years_score = 5

        # ── SIGNAL 2: Employment type quality (max 25 pts) ────────────────
        fulltime_count = sum(1 for s in FULLTIME_SIGNALS if s in lower_text)
        intern_count   = sum(1 for s in INTERN_SIGNALS   if s in lower_text)

        if fulltime_count >= 2:
            employment_score = 25
        elif fulltime_count == 1 and intern_count <= 1:
            employment_score = 20
        elif fulltime_count == 1 and intern_count > 1:
            employment_score = 15
        elif intern_count >= 2:
            employment_score = 10
        elif intern_count == 1:
            employment_score = 7
        else:
            employment_score = 12  # neutral / insufficient signals

        # ── SIGNAL 3: Role title match to JD (max 20 pts) ────────────────
        title_words = [w for w in jd.title.lower().split() if len(w) > 3]
        matched_title_words = sum(
            1 for w in title_words if w in lower_text
        )
        title_match_ratio = matched_title_words / max(len(title_words), 1)

        if title_match_ratio >= 0.7:
            title_score = 20
        elif title_match_ratio >= 0.4:
            title_score = 14
        elif title_match_ratio >= 0.2:
            title_score = 8
        else:
            title_score = 3

        # ── SIGNAL 4: Career progression quality (max 20 pts) ────────────
        progression_count   = sum(1 for s in PROGRESSION_SIGNALS   if s in lower_text)
        company_tier_count  = sum(1 for s in COMPANY_TIER_SIGNALS   if s in lower_text)

        if progression_count >= 4 and company_tier_count >= 2:
            progression_score = 20
        elif progression_count >= 3 or company_tier_count >= 2:
            progression_score = 16
        elif progression_count >= 2 or company_tier_count >= 1:
            progression_score = 12
        elif progression_count >= 1:
            progression_score = 8
        else:
            progression_score = 4

        # ── Combine ───────────────────────────────────────────────────────
        final_experience_score = min(
            100,
            years_score + employment_score + title_score + progression_score,
        )

        logger.info(
            f"Experience breakdown — "
            f"years:{years_score} (source={years_source}, years={years}) "
            f"employment:{employment_score} "
            f"title:{title_score} "
            f"progression:{progression_score} "
            f"total:{final_experience_score}"
        )
        return float(final_experience_score)

    def compute_education_score(self, resume: ResumeEntity, jd: JDEntity) -> float:
        """Grades candidate education level vs the JD's education requirement.

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

        logger.info(f"Education score: {score}")
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
        final = round(final, 2)
        logger.info(
            f"Final score — "
        			# LLM override: use LLM scores directly where available; algorithmic
        			# scores act as fallback only when LLM scores are missing.
            f"ats:{ats_score}×0.20={ats_score*0.20:.1f} "
            f"proj:{project_score}×0.20={project_score*0.20:.1f} "
            f"exp:{experience_score}×0.25={experience_score*0.25:.1f} "
            f"edu:{education_score}×0.10={education_score*0.10:.1f} "
            f"total:{final}"
        )
        return final

    def get_recommendation(self, final_score: float) -> str:
        """
        Returns a hire recommendation string based on final score.
        """
        if final_score >= 65:
            return "Strong Fit"
        elif final_score >= 45:
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

        if word_count >= 250 and sections_found >= 2:
            return "high"
        elif word_count >= 100:
            return "medium"
        return "low"

    def get_confidence(self, all_scores: dict) -> float:
        """
        Computes AI confidence from spread and average of dimension scores.
        Lower confidence when scores are highly uneven or uniformly low.

        Floor reduced to 40.0 (from 60.0) so weak candidates are not
        misleadingly assigned high confidence values.

        Returns a float 0-100.
        """
        import statistics
        scores = list(all_scores.values())
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0

        base_confidence = max(40.0, 100.0 - std_dev)

        avg_score = sum(scores) / len(scores)
        if avg_score < 30:
            base_confidence = min(base_confidence, 55.0)
        elif avg_score < 45:
            base_confidence = min(base_confidence, 70.0)

        logger.info(
            f"Confidence score: {round(base_confidence, 2)} "
            f"(std_dev={round(std_dev, 2)}, avg_score={round(avg_score, 2)})"
        )
        return round(base_confidence, 2)

    # ────────────────────────────────────────────────────────────
    # Orchestration
    # ────────────────────────────────────────────────────────────

    def compute_all(
        self,
        resume: ResumeEntity,
        jd: JDEntity,
        resume_text: str,
        llm_scores: dict | None = None,
    ) -> ScoreResultEntity:
        """
        Runs all scoring dimensions and assembles a complete ScoreResultEntity.

        If llm_scores is provided (keys: 'skills', 'projects', 'experience'),
        each dimension that has an LLM score is blended:
          final_dimension = LLM * 0.60 + algorithmic * 0.40

        Steps:
          1. ATS keyword scoring
          2. Semantic skills scoring  [+ optional LLM blend]
          3. Project quality scoring  [+ optional LLM blend]
          4. Experience scoring       [+ optional LLM blend]
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
            experience_score = self.compute_experience_score(resume, jd, resume_text)

            # LLM blend: 60% LLM + 40% algorithmic for skills / projects / experience
            if llm_scores:
                if llm_scores.get("skills") is not None:
                    algo_skills = skills_score
                    skills_score = round(
                        llm_scores["skills"] * 0.60 + algo_skills * 0.40, 2)
                    logger.info(
                        f"Skills blend: LLM={llm_scores['skills']} "
                        f"algo={algo_skills} final={skills_score}")

                if llm_scores.get("projects") is not None:
                    algo_projects = project_score
                    project_score = round(
                        llm_scores["projects"] * 0.60 + algo_projects * 0.40, 2)
                    logger.info(
                        f"Projects blend: LLM={llm_scores['projects']} "
                        f"algo={algo_projects} final={project_score}")

                if llm_scores.get("experience") is not None:
                    algo_experience = experience_score
                    experience_score = round(
                        llm_scores["experience"] * 0.60 + algo_experience * 0.40, 2)
                    logger.info(
                        f"Experience blend: LLM={llm_scores['experience']} "
                        f"algo={algo_experience} final={experience_score}")

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
            confidence_score = self.get_confidence(all_scores)

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
