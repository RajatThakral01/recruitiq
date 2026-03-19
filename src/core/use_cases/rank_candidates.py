from src.core.ports.storage_port import StoragePort
from src.infrastructure.logger import logger


class RankCandidatesUseCase:
    """
    Retrieves and ranks all screened candidates for a given screening job.
    """

    def __init__(self, storage_port: StoragePort) -> None:
        self.storage_port = storage_port

    def execute(self, job_id: str) -> list[dict]:
        """
        Fetch all candidates for the job, sort by final_score, and enrich
        each entry with resume metadata.

        Args:
            job_id: UUID of the screening job.

        Returns:
            List of candidate result dicts ordered by rank (1 = best).
        """
        logger.info(f"RankCandidatesUseCase: ranking candidates for job_id={job_id}")

        scores = self.storage_port.get_scores_by_job(job_id)

        # Sort descending by final_score (DB already does this, extra safety)
        scores.sort(key=lambda s: s.final_score, reverse=True)

        results = []
        for rank, score in enumerate(scores, start=1):
            # Fetch resume metadata for name and email
            try:
                resume = self.storage_port.get_resume(score.resume_id)
                candidate_name = resume.candidate_name
                email = resume.email
            except Exception:
                candidate_name = "Unknown"
                email = ""

            results.append(
                {
                    "rank": rank,
                    "resume_id": score.resume_id,
                    "candidate_name": candidate_name,
                    "email": email,
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
            )

        logger.info(f"Ranking complete: {len(results)} candidates for job_id={job_id}")
        return results
