import os
import time
import uuid
import tempfile
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from src.adapters.outbound.ocr.pdfplumber_adapter import PDFPlumberAdapter
from src.adapters.outbound.llm.adapter_factory import get_llm_adapter
from src.adapters.outbound.embeddings.sentence_transformer_adapter import SentenceTransformerAdapter
from src.adapters.outbound.storage.postgres_adapter import PostgresAdapter
from src.services.ats_scorer import ATSScorer
from src.services.semantic_scorer import SemanticScorer
from src.services.scoring_engine import ScoringEngine
from src.core.use_cases.screen_resume import ScreenResumeUseCase
from src.core.use_cases.parse_jd import ParseJDUseCase
from src.core.use_cases.rank_candidates import RankCandidatesUseCase
from src.adapters.inbound.api.schemas.resume_schema import (
    ScreenRequest,
    ScreeningResponse,
    CandidateResult,
    UploadJDResponse,
    UploadResumesResponse,
    ErrorResponse,
)
from src.infrastructure import database
from src.infrastructure.config import settings
from src.infrastructure import runtime_config
from src.infrastructure.logger import logger
from src.infrastructure.exceptions import (
    RecruitIQException,
    InvalidFileTypeException,
    FileTooLargeException,
    map_exception_to_http,
)

# ── Dependency instantiation (singletons, module-level) ──────────────────────
pdfplumber_adapter = PDFPlumberAdapter()
llm_adapter = get_llm_adapter()
sentence_transformer_adapter = SentenceTransformerAdapter()
postgres_adapter = PostgresAdapter()

ats_scorer = ATSScorer()
semantic_scorer = SemanticScorer(sentence_transformer_adapter)
scoring_engine = ScoringEngine(ats_scorer, semantic_scorer)

def _get_use_cases():
    """Rebuild use cases with fresh LLM adapter on every request.
    This allows runtime provider switching to take effect immediately.
    """
    llm_adapter = get_llm_adapter()
    return (
        ScreenResumeUseCase(pdfplumber_adapter, llm_adapter, scoring_engine, postgres_adapter),
        ParseJDUseCase(llm_adapter, postgres_adapter),
        llm_adapter,
    )
screen_use_case, parse_jd_use_case, llm_adapter = _get_use_cases()
rank_use_case = RankCandidatesUseCase(postgres_adapter)

# ── Router ────────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/api")


def _get_upload_size_bytes(upload: UploadFile) -> int | None:
    """Best-effort size check without reading file content into memory."""
    try:
        upload.file.seek(0, os.SEEK_END)
        size = upload.file.tell()
        upload.file.seek(0)
        return int(size)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
#  Health
# --------------------------------------------------------------------------- #

@router.post("/config/provider")
def set_provider(payload: dict):
    """Switch LLM provider at runtime for the current session."""
    provider = (payload.get("provider") or "").strip().lower()
    if provider not in {"mistral", "groq"}:
        return JSONResponse(
            status_code=400,
            content={"message": "provider must be one of: mistral, groq", "code": "INVALID_PROVIDER"},
        )
    runtime_config.set_provider(provider)
    logger.info(f"LLM provider switched to: {provider}")
    return {"provider": provider, "status": "switched"}

@router.get("/health")
def health_check():
    """Returns API + database health status."""
    db_ok = database.ping()
    return {
        "status": "ok" if db_ok else "error",
        "db": "connected" if db_ok else "disconnected",
    }


# --------------------------------------------------------------------------- #
#  Upload JD
# --------------------------------------------------------------------------- #

@router.post(
    "/upload/jd",
    response_model=UploadJDResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
)
async def upload_jd(file: UploadFile = File(...)):
    """
    Accept a PDF or plain-text Job Description, extract its text,
    parse it with the configured LLM provider, persist it, and return structured metadata.
    """
    tmp_path = None
    try:
        max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        size_bytes = _get_upload_size_bytes(file)
        if size_bytes is not None and size_bytes > max_bytes:
            raise FileTooLargeException(
                f"{file.filename or 'Uploaded file'} exceeds size limit.",
                detail=f"Maximum allowed size is {settings.MAX_FILE_SIZE_MB} MB.",
            )

        suffix = Path(file.filename).suffix.lower() if file.filename else ".pdf"
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)

        with os.fdopen(tmp_fd, "wb") as tmp_file:
            content = await file.read()
            tmp_file.write(content)

        # Text extraction: PDF → pdfplumber, otherwise decode directly
        if suffix == ".pdf":
            text = pdfplumber_adapter.extract_text(tmp_path)
        else:
            text = content.decode("utf-8", errors="ignore")

        _, parse_jd_use_case_req, _ = _get_use_cases()
        jd = parse_jd_use_case_req.execute(text)

        return UploadJDResponse(
            jd_id=jd.id,
            title=jd.title,
            keywords_extracted=len(jd.keywords),
            required_skills=jd.required_skills,
        )
    except RecruitIQException as e:
        logger.error(f"JD upload error: {e.message}", exc_info=True)
        status, code, message = map_exception_to_http(e)
        return JSONResponse(
            status_code=status,
            content=ErrorResponse(message=message, code=code).model_dump(),
        )
    except Exception as e:
        logger.error(f"Unexpected JD upload error: {e}", exc_info=True)
        status, code, message = map_exception_to_http(e)
        return JSONResponse(
            status_code=status,
            content=ErrorResponse(message=message, code=code).model_dump(),
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# --------------------------------------------------------------------------- #
#  Upload Resumes
# --------------------------------------------------------------------------- #

@router.post(
    "/upload/resumes",
    response_model=UploadResumesResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
)
async def upload_resumes(files: list[UploadFile] = File(...)):
    """
    Accept up to 10 PDF resumes, parse each with the configured LLM provider, persist them,
    and return their IDs.
    """
    from src.core.domain.entities.resume import ResumeEntity

    if len(files) > settings.MAX_RESUMES:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                message=f"Maximum {settings.MAX_RESUMES} resumes allowed per request.",
                code="TOO_MANY_FILES",
            ).model_dump(),
        )

    resume_ids: list[str] = []
    tmp_paths: list[str] = []

    try:
        _, _, llm_adapter_req = _get_use_cases()
        for upload in files:
            max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
            size_bytes = _get_upload_size_bytes(upload)
            if size_bytes is not None and size_bytes > max_bytes:
                raise FileTooLargeException(
                    f"{upload.filename or 'Uploaded file'} exceeds size limit.",
                    detail=f"Maximum allowed size is {settings.MAX_FILE_SIZE_MB} MB.",
                )

            suffix = Path(upload.filename).suffix.lower() if upload.filename else ".pdf"
            if suffix != ".pdf":
                raise InvalidFileTypeException(
                    f"{upload.filename} is not a PDF.", detail="Only .pdf files are supported."
                )

            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
            tmp_paths.append(tmp_path)

            with os.fdopen(tmp_fd, "wb") as tmp_file:
                tmp_file.write(await upload.read())

            text = pdfplumber_adapter.extract_text(tmp_path)
            quality_flag = scoring_engine.get_quality_flag(text)
            parsed = llm_adapter_req.parse_resume(text)

            resume = ResumeEntity(
                id=str(uuid.uuid4()),
                candidate_name=parsed.get("candidate_name", ""),
                email=parsed.get("email") or "",
                raw_text=text,
                parsed_skills=parsed.get("parsed_skills", []),
                years_experience=float(parsed.get("years_experience", 0.0)),
                relevant_experience=float(parsed.get("relevant_experience", 0.0)),
                education_level=parsed.get("education_level", "Other"),
                projects=parsed.get("projects", []),
                quality_flag=quality_flag,
                created_at=datetime.now(),
            )
            inserted_id = postgres_adapter.save_resume(resume)
            resume_ids.append(inserted_id)

        return UploadResumesResponse(
            resume_ids=resume_ids,
            files_processed=len(resume_ids),
        )
    except RecruitIQException as e:
        logger.error(f"Resume upload error: {e.message}", exc_info=True)
        status, code, message = map_exception_to_http(e)
        return JSONResponse(
            status_code=status,
            content=ErrorResponse(message=message, code=code).model_dump(),
        )
    except Exception as e:
        logger.error(f"Unexpected resume upload error: {e}", exc_info=True)
        status, code, message = map_exception_to_http(e)
        return JSONResponse(
            status_code=status,
            content=ErrorResponse(message=message, code=code).model_dump(),
        )
    finally:
        for p in tmp_paths:
            if os.path.exists(p):
                os.remove(p)


# --------------------------------------------------------------------------- #
#  Screen
# --------------------------------------------------------------------------- #

@router.post(
    "/screen",
    response_model=ScreeningResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
)
async def screen_resumes(request: ScreenRequest):
    """
    Trigger the full screening pipeline for a list of resumes against a JD.
    Returns ranked results when complete.
    """
    job_id = None
    try:
        _, _, llm_adapter_req = _get_use_cases()
        jd = postgres_adapter.get_jd(request.jd_id)
        job_id = postgres_adapter.create_screening_job(request.jd_id)
        logger.info(f"Screening job {job_id} created for JD {request.jd_id}")

        failed_resumes: list[str] = []
        successful_count = 0

        for resume_id in request.resume_ids:
            try:
                resume_start = time.perf_counter()
                resume = postgres_adapter.get_resume(resume_id)

                llm_scores = {
                    "skills": None,
                    "projects": None,
                    "experience": None,
                }
                scoring_mode = (settings.SCORING_MODE or "legacy").strip().lower()
                if scoring_mode in {"hybrid", "llm_first"}:
                    try:
                        llm_eval = llm_adapter_req.evaluate_all_parallel(
                            resume.raw_text,
                            jd.to_dict(),
                        )
                        llm_scores = {
                            "skills": llm_eval.get("skills_score"),
                            "projects": llm_eval.get("project_score"),
                            "experience": llm_eval.get("experience_score"),
                        }
                    except Exception as llm_error:
                        logger.warning(
                            f"LLM parallel evaluation failed for resume {resume_id}; "
                            f"falling back to algorithmic-only dimensions: {llm_error}"
                        )

                # Bypass OCR for already-stored resumes: extract text directly
                result = scoring_engine.compute_all(
                    resume,
                    jd,
                    resume.raw_text,
                    llm_scores=llm_scores,
                )

                # Set the job_id on the result
                result.job_id = job_id

                sg = llm_adapter_req.generate_strengths_gaps(
                    resume.to_dict(),
                    jd.to_dict(),
                    {
                        "final_score": result.final_score,
                        "skills_score": result.skills_score,
                        "ats_score": result.ats_score,
                    },
                )
                result.strengths = sg.get("strengths", [])
                result.gaps = sg.get("gaps", [])
                result.processing_time_seconds = round(time.perf_counter() - resume_start, 3)

                postgres_adapter.save_score(result)
                successful_count += 1
            except Exception as e:
                logger.error(
                    f"Failed to screen resume {resume_id}: {e}",
                    exc_info=True,
                )
                failed_resumes.append(resume_id)
                continue  # Skip failed resume; continue with others

        if successful_count == 0:
            msg = (
                f"Failed to screen all resumes for job {job_id}. "
                f"Resume IDs: {failed_resumes}"
            )
            logger.error(msg)
            raise RuntimeError(msg)

        if failed_resumes:
            logger.warning(
                f"Screening completed with failures for job {job_id}. "
                f"Failed resume IDs: {failed_resumes}"
            )

        postgres_adapter.update_job_status(job_id, "complete")
        ranked = rank_use_case.execute(job_id)

        return ScreeningResponse(
            job_id=job_id,
            status="complete",
            jd_title=jd.title,
            total_screened=len(ranked),
            results=[CandidateResult(**r) for r in ranked],
        )
    except Exception as e:
        logger.error(f"Screening pipeline error: {e}", exc_info=True)
        if job_id:
            postgres_adapter.update_job_status(job_id, "failed")
        status, code, message = map_exception_to_http(e)
        return JSONResponse(
            status_code=status,
            content=ErrorResponse(message=message, code=code).model_dump(),
        )


# --------------------------------------------------------------------------- #
#  Results
# --------------------------------------------------------------------------- #

@router.get(
    "/results/{job_id}",
    response_model=ScreeningResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def get_results(job_id: str):
    """Retrieve ranked results for a previously submitted screening job."""
    try:
        ranked = rank_use_case.execute(job_id)
        # Try to recover JD title from first result's stored score
        jd_title = ""
        scores = postgres_adapter.get_scores_by_job(job_id)
        if not scores and not postgres_adapter.job_exists(job_id):
            return JSONResponse(
                status_code=404,
                content={
                    "error": True,
                    "message": f"Job {job_id} not found",
                    "code": "JOB_NOT_FOUND",
                },
            )
        if scores:
            try:
                jd = postgres_adapter.get_jd(scores[0].jd_id)
                jd_title = jd.title
            except Exception:
                pass

        job_status = postgres_adapter.get_job_status(job_id) or "unknown"
        return ScreeningResponse(
            job_id=job_id,
            status=job_status,
            jd_title=jd_title,
            total_screened=len(ranked),
            results=[CandidateResult(**r) for r in ranked],
        )
    except RecruitIQException as e:
        logger.error(f"Results retrieval error: {e.message}", exc_info=True)
        status, code, message = map_exception_to_http(e)
        return JSONResponse(
            status_code=status,
            content=ErrorResponse(message=message, code=code).model_dump(),
        )
    except Exception as e:
        logger.error(f"Results retrieval error: {e}", exc_info=True)
        status, code, message = map_exception_to_http(e)
        return JSONResponse(
            status_code=status,
            content=ErrorResponse(message=message, code=code).model_dump(),
        )


@router.get("/results/{job_id}/candidate/{resume_id}")
def get_candidate_result(job_id: str, resume_id: str):
    """Retrieve the full profile for one candidate from a screening job."""
    try:
        scores = postgres_adapter.get_scores_by_job(job_id)
        score = next((s for s in scores if str(s.resume_id) == resume_id), None)
        if not score:
            raise HTTPException(status_code=404, detail="Candidate not found in this job.")

        resume = postgres_adapter.get_resume(resume_id)
        return {
            "resume_id": resume_id,
            "candidate_name": resume.candidate_name,
            "email": resume.email,
            "parsed_skills": resume.parsed_skills,
            "years_experience": resume.years_experience,
            "education_level": resume.education_level,
            "projects": resume.projects,
            "final_score": score.final_score,
            "skills_score": score.skills_score,
            "ats_score": score.ats_score,
            "project_score": score.project_score,
            "experience_score": score.experience_score,
            "education_score": score.education_score,
            "strengths": score.strengths,
            "gaps": score.gaps,
            "matched_keywords": score.matched_keywords,
            "missing_keywords": score.missing_keywords,
            "keyword_match_rate": score.keyword_match_rate,
            "confidence_score": score.confidence_score,
            "quality_flag": score.quality_flag,
            "recommendation": score.recommendation,
            "processing_time_seconds": score.processing_time_seconds,
        }
    except HTTPException as e:
        status, code, message = map_exception_to_http(e)
        return JSONResponse(
            status_code=status,
            content=ErrorResponse(message=message, code=code).model_dump(),
        )
    except RecruitIQException as e:
        logger.error(f"Candidate result retrieval error: {e.message}", exc_info=True)
        status, code, message = map_exception_to_http(e)
        return JSONResponse(
            status_code=status,
            content=ErrorResponse(message=message, code=code).model_dump(),
        )
    except Exception as e:
        logger.error(f"Candidate result retrieval error: {e}", exc_info=True)
        status, code, message = map_exception_to_http(e)
        return JSONResponse(
            status_code=status,
            content=ErrorResponse(message=message, code=code).model_dump(),
        )
