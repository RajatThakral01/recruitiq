import time
import uuid
from datetime import datetime

from src.core.domain.entities.resume import ResumeEntity
from src.core.domain.entities.job_description import JDEntity
from src.core.domain.entities.score_result import ScoreResultEntity
from src.core.ports.ocr_port import OCRPort
from src.core.ports.llm_port import LLMPort
from src.core.ports.storage_port import StoragePort
from src.services.scoring_engine import ScoringEngine
from src.infrastructure.config import settings
from src.infrastructure.logger import logger
from src.infrastructure.exceptions import InvalidFileTypeException, LLMAuthenticationException


class ScreenResumeUseCase:
    """
    Orchestrates the full resume screening pipeline for a single resume.
    """

    def __init__(
        self,
        ocr_port: OCRPort,
        llm_port: LLMPort,
        scoring_engine: ScoringEngine,
        storage_port: StoragePort,
    ) -> None:
        self.ocr_port = ocr_port
        self.llm_port = llm_port
        self.scoring_engine = scoring_engine
        self.storage_port = storage_port

    def execute(
        self,
        file_path: str,
        jd: JDEntity,
        job_id: str,
    ) -> ScoreResultEntity:
        """
        Run the full screening pipeline for one resume against a JD.

        Args:
            file_path: Absolute path to the uploaded PDF file.
            jd: Parsed job description entity.
            job_id: UUID of the screening job this resume belongs to.

        Returns:
            A fully populated ScoreResultEntity including strengths/gaps.

        Raises:
            InvalidFileTypeException: If file validation fails.
        """
        pipeline_start = time.time()

        # 1. Validate file
        logger.info(f"ScreenResumeUseCase: validating file {file_path}")
        if not self.ocr_port.validate_file(file_path):
            raise InvalidFileTypeException(
                f"File validation failed for {file_path}."
            )

        # 2. Extract text via OCR
        logger.info("ScreenResumeUseCase: extracting text via OCR.")
        resume_text = self.ocr_port.extract_text(file_path)

        # 3. Determine quality flag early (based on raw text)
        quality_flag = self.scoring_engine.get_quality_flag(resume_text)

        # 4. Parse resume structure with LLM
        logger.info("ScreenResumeUseCase: parsing resume with LLM.")
        parsed = self.llm_port.parse_resume(resume_text)

        # 5. Build ResumeEntity
        resume = ResumeEntity(
            id=str(uuid.uuid4()),
            candidate_name=parsed.get("candidate_name", ""),
            email=parsed.get("email") or "",
            raw_text=resume_text,
            parsed_skills=parsed.get("parsed_skills", []),
            years_experience=float(parsed.get("years_experience", 0.0)),
            relevant_experience=float(parsed.get("relevant_experience", 0.0)),
            education_level=parsed.get("education_level", "Other"),
            projects=parsed.get("projects", []),
            quality_flag=quality_flag,
            created_at=datetime.now(),
        )

        # 6. Persist resume
        logger.info(f"ScreenResumeUseCase: saving resume for {resume.candidate_name}")
        self.storage_port.save_resume(resume)

        # 7. Optional LLM evaluation by scoring mode
        scoring_mode = (settings.SCORING_MODE or "legacy").strip().lower()
        llm_scores = {
            "skills": None,
            "projects": None,
            "experience": None,
        }
        if scoring_mode in {"hybrid", "llm_first"}:
            logger.info("Step 7/8 — Parallel LLM scoring")
            try:
                llm_eval = self.llm_port.evaluate_all_parallel(
                    resume_text, jd.to_dict()
                )
                llm_scores = {
                    "skills": llm_eval.get("skills_score"),
                    "projects": llm_eval.get("project_score"),
                    "experience": llm_eval.get("experience_score"),
                }
                logger.info(
                    f"LLM scores — "
                    f"skills:{llm_scores['skills']} "
                    f"projects:{llm_scores['projects']} "
                    f"experience:{llm_scores['experience']}"
                )
            except LLMAuthenticationException as e:
                logger.warning(
                    f"LLM auth failed during parallel scoring; "
                    f"using algorithmic scoring fallback for this request: {e}"
                )
                llm_scores = {
                    "skills": None,
                    "projects": None,
                    "experience": None,
                }
            except Exception as e:
                logger.warning(
                    f"Parallel LLM scoring failed, using algorithmic fallback: {e}"
                )
        else:
            logger.info("Step 7/8 — Parallel LLM scoring skipped (legacy mode)")

        # 8. Run all scoring dimensions (pure LLM scores for semantic dimensions)
        logger.info("ScreenResumeUseCase: running scoring engine.")
        result: ScoreResultEntity = self.scoring_engine.compute_all(
            resume, jd, resume_text, llm_scores=llm_scores
        )

        # 9. Generate strengths and gaps via LLM
        logger.info("ScreenResumeUseCase: generating strengths and gaps.")
        try:
            sg = self.llm_port.generate_strengths_gaps(
                resume.to_dict(),
                jd.to_dict(),
                {
                    "final_score": result.final_score,
                    "skills_score": result.skills_score,
                    "ats_score": result.ats_score,
                },
            )
        except LLMAuthenticationException as e:
            logger.warning(
                f"LLM auth failed during strengths/gaps generation; "
                f"continuing without strengths/gaps: {e}"
            )
            sg = {"strengths": [], "gaps": []}

        # 10. Patch result with LLM-generated fields
        result.strengths = sg.get("strengths", [])
        result.gaps = sg.get("gaps", [])
        result.job_id = job_id
        result.processing_time_seconds = round(time.time() - pipeline_start, 2)

        # 11. Persist score result
        try:
            self.storage_port.save_score(result)
            logger.info(f"ScreenResumeUseCase: score saved successfully for resume_id={resume.id}")
        except Exception as e:
            logger.error(f"ScreenResumeUseCase: failed to save score for resume_id={resume.id}: {e}")
            raise

        logger.info(
            f"Resume screened: {resume.candidate_name} — "
            f"score={result.final_score} ({result.recommendation})"
        )
        logger.info(
            f"Total pipeline time for "
            f"{resume.candidate_name}: "
            f"{time.time() - pipeline_start:.1f}s"
        )
        return result
