import json
import uuid
from datetime import datetime
from typing import List

from src.core.domain.entities.resume import ResumeEntity
from src.core.domain.entities.job_description import JDEntity
from src.core.domain.entities.score_result import ScoreResultEntity
from src.core.ports.storage_port import StoragePort
from src.infrastructure.database import get_connection
from src.infrastructure.logger import logger
from src.infrastructure.exceptions import DatabaseException


class PostgresAdapter(StoragePort):
    """
    Concrete StoragePort implementation using the psycopg2 ThreadedConnectionPool
    from infrastructure.database.
    """

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _j(value) -> str:
        """Serialize Python list/dict to JSON string for JSONB columns.
        
        Handles datetime objects by converting them to ISO format strings.
        """
        def _serialize(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_serialize(item) for item in obj]
            return obj
        
        return json.dumps(_serialize(value))

    # ── Resume ────────────────────────────────────────────────────────────────

    def save_resume(self, resume: ResumeEntity) -> str:
        """Persist a ResumeEntity to the resumes table. Returns the inserted UUID."""
        sql = """
            INSERT INTO resumes (
                id, candidate_name, email, raw_text,
                parsed_skills, years_experience, education_level,
                projects, quality_flag, created_at
            ) VALUES (
                %(id)s, %(candidate_name)s, %(email)s, %(raw_text)s,
                %(parsed_skills)s::jsonb, %(years_experience)s, %(education_level)s,
                %(projects)s::jsonb, %(quality_flag)s, %(created_at)s
            )
            ON CONFLICT (id) DO NOTHING
            RETURNING id;
        """
        params = {
            "id": resume.id or str(uuid.uuid4()),
            "candidate_name": resume.candidate_name,
            "email": resume.email,
            "raw_text": resume.raw_text,
            "parsed_skills": self._j(resume.parsed_skills),
            "years_experience": resume.years_experience,
            "education_level": resume.education_level,
            "projects": self._j(resume.projects),
            "quality_flag": resume.quality_flag,
            "created_at": resume.created_at.isoformat() if hasattr(resume.created_at, "isoformat") else str(resume.created_at),
        }
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    row = cur.fetchone()
                    inserted_id = str(row[0]) if row else params["id"]
            logger.info(f"Resume saved: id={inserted_id}")
            return inserted_id
        except Exception as e:
            logger.error(f"save_resume failed: {e}")
            raise DatabaseException("Could not save resume.", detail=str(e))

    def get_resume(self, resume_id: str) -> ResumeEntity:
        """Fetch a single resume by UUID."""
        sql = "SELECT id, candidate_name, email, raw_text, parsed_skills, years_experience, education_level, projects, quality_flag, created_at FROM resumes WHERE id = %s;"
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (resume_id,))
                    row = cur.fetchone()
            if not row:
                raise DatabaseException(f"Resume {resume_id} not found.")
            return ResumeEntity(
                id=str(row[0]),
                candidate_name=row[1] or "",
                email=row[2] or "",
                raw_text=row[3] or "",
                parsed_skills=row[4] if isinstance(row[4], list) else json.loads(row[4] or "[]"),
                years_experience=float(row[5] or 0.0),
                education_level=row[6] or "Other",
                projects=row[7] if isinstance(row[7], list) else json.loads(row[7] or "[]"),
                quality_flag=row[8] or "medium",
                created_at=row[9] if isinstance(row[9], datetime) else datetime.now(),
            )
        except DatabaseException:
            raise
        except Exception as e:
            logger.error(f"get_resume failed: {e}")
            raise DatabaseException("Could not retrieve resume.", detail=str(e))

    # ── Job Description ───────────────────────────────────────────────────────

    def save_jd(self, jd: JDEntity) -> str:
        """Persist a JDEntity to the job_descriptions table. Returns the inserted UUID."""
        sql = """
            INSERT INTO job_descriptions (
                id, title, required_skills, preferred_skills,
                min_experience, education_requirement, keywords,
                must_have, nice_to_have, seniority_level, raw_text, created_at
            ) VALUES (
                %(id)s, %(title)s, %(required_skills)s::jsonb,
                %(preferred_skills)s::jsonb, %(min_experience)s,
                %(education_requirement)s, %(keywords)s::jsonb,
                %(must_have)s::jsonb, %(nice_to_have)s::jsonb,
                %(seniority_level)s, %(raw_text)s, %(created_at)s
            )
            ON CONFLICT (id) DO NOTHING
            RETURNING id;
        """
        params = {
            "id": jd.id or str(uuid.uuid4()),
            "title": jd.title,
            "required_skills": self._j(jd.required_skills),
            "preferred_skills": self._j(jd.preferred_skills),
            "min_experience": jd.min_experience,
            "education_requirement": jd.education_requirement,
            "keywords": self._j(jd.keywords),
            "must_have": self._j(jd.must_have),
            "nice_to_have": self._j(jd.nice_to_have),
            "seniority_level": jd.seniority_level,
            "raw_text": jd.raw_text,
            "created_at": jd.created_at.isoformat() if hasattr(jd.created_at, "isoformat") else str(jd.created_at),
        }
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    row = cur.fetchone()
                    inserted_id = str(row[0]) if row else params["id"]
            logger.info(f"JD saved: id={inserted_id}, title={jd.title}")
            return inserted_id
        except Exception as e:
            logger.error(f"save_jd failed: {e}")
            raise DatabaseException("Could not save job description.", detail=str(e))

    def get_jd(self, jd_id: str) -> JDEntity:
        """Fetch a single job description by UUID."""
        sql = "SELECT id, title, required_skills, preferred_skills, keywords, must_have, nice_to_have, seniority_level, min_experience, education_requirement, raw_text, created_at FROM job_descriptions WHERE id = %s;"
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (jd_id,))
                    row = cur.fetchone()
            if not row:
                raise DatabaseException(f"Job description {jd_id} not found.")
            def _loads(v):
                return v if isinstance(v, list) else json.loads(v or "[]")
            return JDEntity(
                id=str(row[0]),
                title=row[1] or "",
                required_skills=_loads(row[2]),
                preferred_skills=_loads(row[3]),
                keywords=_loads(row[4]),
                must_have=_loads(row[5]),
                nice_to_have=_loads(row[6]),
                seniority_level=row[7] or "junior",
                min_experience=float(row[8] or 0.0),
                education_requirement=row[9] or "Other",
                raw_text=row[10] or "",
                created_at=row[11] if isinstance(row[11], datetime) else datetime.now(),
            )
        except DatabaseException:
            raise
        except Exception as e:
            logger.error(f"get_jd failed: {e}")
            raise DatabaseException("Could not retrieve job description.", detail=str(e))

    # ── Score Results ─────────────────────────────────────────────────────────

    def save_score(self, score: ScoreResultEntity) -> str:
        """Persist a ScoreResultEntity to score_results. Returns inserted UUID."""
        sql = """
            INSERT INTO score_results (
                id, job_id, resume_id, jd_id,
                skills_score, ats_score, project_score,
                experience_score, education_score, final_score,
                strengths, gaps, matched_keywords, missing_keywords,
                keyword_match_rate, confidence_score, quality_flag,
                recommendation, processing_time_seconds, created_at
            ) VALUES (
                %(id)s, %(job_id)s, %(resume_id)s, %(jd_id)s,
                %(skills_score)s, %(ats_score)s, %(project_score)s,
                %(experience_score)s, %(education_score)s, %(final_score)s,
                %(strengths)s::jsonb, %(gaps)s::jsonb,
                %(matched_keywords)s::jsonb, %(missing_keywords)s::jsonb,
                %(keyword_match_rate)s, %(confidence_score)s, %(quality_flag)s,
                %(recommendation)s, %(processing_time_seconds)s, %(created_at)s
            )
            ON CONFLICT (id) DO NOTHING
            RETURNING id;
        """
        params = {
            "id": score.id or str(uuid.uuid4()),
            "job_id": score.job_id,
            "resume_id": score.resume_id,
            "jd_id": score.jd_id,
            "skills_score": score.skills_score,
            "ats_score": score.ats_score,
            "project_score": score.project_score,
            "experience_score": score.experience_score,
            "education_score": score.education_score,
            "final_score": score.final_score,
            "strengths": self._j(score.strengths),
            "gaps": self._j(score.gaps),
            "matched_keywords": self._j(score.matched_keywords),
            "missing_keywords": self._j(score.missing_keywords),
            "keyword_match_rate": score.keyword_match_rate,
            "confidence_score": score.confidence_score,
            "quality_flag": score.quality_flag,
            "recommendation": score.recommendation,
            "processing_time_seconds": score.processing_time_seconds,
            "created_at": score.created_at.isoformat() if hasattr(score.created_at, "isoformat") else str(score.created_at),
        }
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    row = cur.fetchone()
                    inserted_id = str(row[0]) if row else params["id"]
            logger.info(f"Score saved: id={inserted_id}, final_score={score.final_score}")
            return inserted_id
        except Exception as e:
            logger.error(f"save_score failed: {e}")
            raise DatabaseException("Could not save score result.", detail=str(e))

    def get_scores_by_job(self, job_id: str) -> List[ScoreResultEntity]:
        """Return all score results for a job, ordered by final_score DESC."""
        sql = """
            SELECT id, job_id, resume_id, jd_id,
                   skills_score, ats_score, project_score,
                   experience_score, education_score, final_score,
                   strengths, gaps, matched_keywords, missing_keywords,
                   keyword_match_rate, confidence_score, quality_flag,
                   recommendation, processing_time_seconds, created_at
            FROM score_results
            WHERE job_id = %s
            ORDER BY final_score DESC;
        """
        def _loads(v):
            return v if isinstance(v, list) else json.loads(v or "[]")

        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (job_id,))
                    rows = cur.fetchall()

            results = []
            for row in rows:
                sr = ScoreResultEntity(
                    id=str(row[0]),
                    job_id=str(row[1]),
                    resume_id=str(row[2]),
                    jd_id=str(row[3]),
                    skills_score=float(row[4] or 0),
                    ats_score=float(row[5] or 0),
                    project_score=float(row[6] or 0),
                    experience_score=float(row[7] or 0),
                    education_score=float(row[8] or 0),
                    final_score=float(row[9] or 0),
                    strengths=_loads(row[10]),
                    gaps=_loads(row[11]),
                    matched_keywords=_loads(row[12]),
                    missing_keywords=_loads(row[13]),
                    keyword_match_rate=float(row[14] or 0),
                    confidence_score=float(row[15] or 0),
                    quality_flag=row[16] or "medium",
                    recommendation=row[17] or "Not Fit",
                    processing_time_seconds=float(row[18] or 0),
                    created_at=row[19] if isinstance(row[19], datetime) else datetime.now(),
                )
                results.append(sr)

            logger.info(f"Retrieved {len(results)} scores for job_id={job_id}")
            return results
        except Exception as e:
            logger.error(f"get_scores_by_job failed: {e}")
            raise DatabaseException("Could not retrieve scores.", detail=str(e))

    # ── Screening Jobs ────────────────────────────────────────────────────────

    def create_screening_job(self, jd_id: str) -> str:
        """Insert a new screening job with status='pending'. Returns job UUID."""
        sql = """
            INSERT INTO screening_jobs (jd_id, status)
            VALUES (%s, 'pending')
            RETURNING id;
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (jd_id,))
                    job_id = str(cur.fetchone()[0])
            logger.info(f"Screening job created: id={job_id}")
            return job_id
        except Exception as e:
            logger.error(f"create_screening_job failed: {e}")
            raise DatabaseException("Could not create screening job.", detail=str(e))

    def update_job_status(self, job_id: str, status: str) -> None:
        """Update job status; sets completed_at when status is 'complete'."""
        sql = """
            UPDATE screening_jobs
            SET status = %s,
                completed_at = CASE
                    WHEN %s = 'complete' THEN NOW()
                    ELSE completed_at
                END
            WHERE id = %s;
        """
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (status, status, job_id))
            logger.info(f"Job {job_id} status → '{status}'")
        except Exception as e:
            logger.error(f"update_job_status failed: {e}")
            raise DatabaseException("Could not update job status.", detail=str(e))
