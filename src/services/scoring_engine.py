import uuid
import time
from datetime import datetime
from src.core.domain.entities.resume import ResumeEntity
from src.core.domain.entities.job_description import JDEntity
from src.core.domain.entities.score_result import ScoreResultEntity
from src.services.ats_scorer import ATSScorer
from src.services.semantic_scorer import SemanticScorer
from src.infrastructure.config import settings
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

    def _resolve_dimension_score(
        self,
        *,
        dimension: str,
        algorithmic_score: float,
        llm_score: float | None,
        scoring_mode: str,
    ) -> float:
        """Resolve final score for a dimension according to scoring mode.

        Modes:
          - legacy: keep existing algorithmic score only.
          - llm_first: prefer LLM score whenever available.
          - hybrid: blend 60% LLM + 40% algorithmic when available.
        """
        if scoring_mode == "legacy":
            resolved = round(float(algorithmic_score), 2)
            logger.info(
            f"{dimension} mode=legacy: algo={algorithmic_score} final={resolved}"
            )
            return resolved

        if llm_score is None:
            resolved = round(float(algorithmic_score), 2)
            logger.info(
                f"{dimension} mode={scoring_mode}: llm_missing algo={algorithmic_score} final={resolved}"
            )
            return resolved

        if scoring_mode == "llm_first":
            resolved = round(float(llm_score), 2)
            logger.info(
                f"{dimension} mode=llm_first: llm={llm_score} algo={algorithmic_score} final={resolved}"
            )
            return resolved

        if dimension == "skills":
            resolved = round(float(llm_score) * 0.80 + float(algorithmic_score) * 0.20, 2)
        else:
            resolved = round(float(llm_score) * 0.65 + float(algorithmic_score) * 0.35, 2)
        logger.info(
            f"{dimension} mode={scoring_mode}: llm={llm_score} algo={algorithmic_score} final={resolved}"
        )
        return resolved

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

        # ── SIGNAL 1: Years of relevant experience (max 30 pts) ───────────
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
            years_score = 30  # no minimum set — full marks
        elif years >= required * 1.5:
            years_score = 30
        elif years >= required:
            years_score = 24
        elif years >= required * 0.7:
            years_score = 15
        elif years >= required * 0.5:
            years_score = 8
        else:
            years_score = 5

        # ── SIGNAL 2: Employment type quality (max 20 pts) ────────────────
        fulltime_count = sum(1 for s in FULLTIME_SIGNALS if s in lower_text)
        intern_count   = sum(1 for s in INTERN_SIGNALS   if s in lower_text)

        # Relevant internships should contribute meaningfully, but less than full-time roles.
        if fulltime_count >= 2:
            employment_score = 20
        elif fulltime_count == 1 and intern_count <= 1:
            employment_score = 16
        elif fulltime_count == 1 and intern_count > 1:
            employment_score = 14
        elif intern_count >= 2:
            employment_score = 11
        elif intern_count == 1:
            employment_score = 8
        else:
            employment_score = 10  # neutral / insufficient signals

        # ── SIGNAL 3: Role + skill relevance to JD (max 25 pts) ───────────
        title_words = [w for w in jd.title.lower().split() if len(w) > 3]
        matched_title_words = sum(
            1 for w in title_words if w in lower_text
        )
        title_match_ratio = matched_title_words / max(len(title_words), 1)

        required_skill_terms = [s.strip().lower() for s in jd.required_skills if s.strip()]
        matched_required_skill_terms = sum(1 for s in required_skill_terms if s in lower_text)
        required_skill_ratio = matched_required_skill_terms / max(len(required_skill_terms), 1)

        relevance_ratio = (0.55 * title_match_ratio) + (0.45 * required_skill_ratio)
        if relevance_ratio >= 0.7:
            role_relevance_score = 25
        elif relevance_ratio >= 0.5:
            role_relevance_score = 20
        elif relevance_ratio >= 0.35:
            role_relevance_score = 14
        elif relevance_ratio >= 0.2:
            role_relevance_score = 8
        else:
            role_relevance_score = 4

        # ── SIGNAL 4: Company context and role environment (max 15 pts) ───
        company_tier_count = sum(1 for s in COMPANY_TIER_SIGNALS if s in lower_text)
        if company_tier_count >= 3:
            company_context_score = 15
        elif company_tier_count >= 2:
            company_context_score = 12
        elif company_tier_count >= 1:
            company_context_score = 9
        else:
            # If explicit company signals are absent, use role relevance as proxy context.
            company_context_score = 7 if relevance_ratio >= 0.45 else 5

        # ── SIGNAL 5: Career progression quality (max 10 pts) ─────────────
        progression_count   = sum(1 for s in PROGRESSION_SIGNALS   if s in lower_text)

        if progression_count >= 4:
            progression_score = 10
        elif progression_count >= 3:
            progression_score = 8
        elif progression_count >= 2:
            progression_score = 6
        elif progression_count >= 1:
            progression_score = 4
        else:
            progression_score = 2

        # ── Combine ───────────────────────────────────────────────────────
        final_experience_score = min(
            100,
            years_score + employment_score + role_relevance_score + company_context_score + progression_score,
        )

        logger.info(
            f"Experience breakdown — "
            f"years:{years_score} (source={years_source}, years={years}) "
            f"employment:{employment_score} "
            f"role_relevance:{role_relevance_score} "
            f"company_context:{company_context_score} "
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
            f"skills:{skills_score}×{WEIGHTS['skills']:.2f}={skills_score*WEIGHTS['skills']:.1f} "
            f"ats:{ats_score}×{WEIGHTS['ats']:.2f}={ats_score*WEIGHTS['ats']:.1f} "
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
        if final_score >= 75:
            return "Strong Fit"
        elif final_score >= 55:
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
                dimension resolution is based on SCORING_MODE:
                    - llm_first: use LLM score directly when available
                    - hybrid / legacy: blend LLM 60% + algorithmic 40%

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
            pipeline_start = time.perf_counter()
            logger.info(
                f"Starting full scoring pipeline for resume_id={resume.id}, jd_id={jd.id}"
            )

            scoring_mode = (settings.SCORING_MODE or "legacy").strip().lower()
            if scoring_mode not in {"legacy", "hybrid", "llm_first"}:
                logger.warning(f"Unknown SCORING_MODE='{scoring_mode}', falling back to legacy")
                scoring_mode = "legacy"

            # 1. ATS
            logger.info("Step 1/8 — ATS scoring")
            ats_result = self.ats_scorer.score(resume_text, jd)
            ats_score = ats_result["ats_score"]
            matched_keywords = ats_result["matched_keywords"]
            missing_keywords = ats_result["missing_keywords"]
            keyword_match_rate = ats_result["keyword_match_rate"]

            llm_scores = llm_scores or {}
            warning_flags: dict[str, str] = {}

            needs_algo_skills = not (
                scoring_mode == "llm_first" and llm_scores.get("skills") is not None
            )
            needs_algo_projects = not (
                scoring_mode == "llm_first" and llm_scores.get("projects") is not None
            )
            needs_algo_experience = not (
                scoring_mode == "llm_first" and llm_scores.get("experience") is not None
            )

            # 2. Semantic skills
            if needs_algo_skills:
                logger.info("Step 2/8 — Semantic skills scoring")
                skills_score = self.semantic_scorer.score_skills(resume_text, jd)
            else:
                logger.info("Step 2/8 — Semantic skills scoring skipped (llm_first)")
                skills_score = 50.0

            # 3. Project quality
            if needs_algo_projects:
                logger.info("Step 3/8 — Project quality scoring")
                project_score = self.semantic_scorer.score_projects(resume_text, jd)
            else:
                logger.info("Step 3/8 — Project quality scoring skipped (llm_first)")
                project_score = 40.0

            # 4. Experience
            if needs_algo_experience:
                logger.info("Step 4/8 — Experience scoring")
                experience_score = self.compute_experience_score(resume, jd, resume_text)
            else:
                logger.info("Step 4/8 — Experience scoring skipped (llm_first)")
                experience_score = 40.0

            # Resolve LLM-influenced dimensions according to configured scoring mode
            if llm_scores:
                if scoring_mode in {"hybrid", "llm_first"}:
                    for dimension_key in ("skills", "projects", "experience"):
                        if llm_scores.get(dimension_key) is None:
                            warning_flags[f"{dimension_key}_llm_fallback"] = (
                                "Used algorithmic fallback because LLM score was unavailable."
                            )
                            logger.warning(
                                f"fallback_trigger dimension={dimension_key} mode={scoring_mode} reason=llm_score_unavailable"
                            )
                skills_score = self._resolve_dimension_score(
                    dimension="skills",
                    algorithmic_score=skills_score,
                    llm_score=llm_scores.get("skills"),
                    scoring_mode=scoring_mode,
                )
                project_score = self._resolve_dimension_score(
                    dimension="projects",
                    algorithmic_score=project_score,
                    llm_score=llm_scores.get("projects"),
                    scoring_mode=scoring_mode,
                )
                experience_score = self._resolve_dimension_score(
                    dimension="experience",
                    algorithmic_score=experience_score,
                    llm_score=llm_scores.get("experience"),
                    scoring_mode=scoring_mode,
                )
            elif scoring_mode in {"hybrid", "llm_first"}:
                warning_flags["llm_scores_unavailable"] = (
                    "LLM dimension outputs were unavailable; algorithmic fallback used for all semantic dimensions."
                )
                logger.warning(
                    f"fallback_trigger dimension=all mode={scoring_mode} reason=llm_scores_unavailable"
                )

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

            total_latency = round(time.perf_counter() - pipeline_start, 3)
            logger.info(
                f"scoring_pipeline_complete latency_seconds={total_latency} mode={scoring_mode} "
                f"warning_count={len(warning_flags)}"
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
                warning_flags=warning_flags,
                llm_provider=settings.LLM_PROVIDER if scoring_mode in {"hybrid", "llm_first"} else "",
                llm_model=(
                    settings.MISTRAL_MODEL if settings.LLM_PROVIDER == "mistral" else settings.GROQ_MODEL
                )
                if scoring_mode in {"hybrid", "llm_first"}
                else "",
                prompt_version="scoring-prompts-v1"
                if scoring_mode in {"hybrid", "llm_first"}
                else "",
                processing_time_seconds=0.0,  # set by the calling use case
                created_at=datetime.now(),
            )

        except ScoringException:
            raise
        except Exception as e:
            logger.error(f"Scoring pipeline failed: {str(e)}", exc_info=True)
            raise ScoringException("Full scoring pipeline failed.", detail=str(e))
