import re
from rapidfuzz import fuzz
from src.core.domain.entities.job_description import JDEntity
from src.infrastructure.logger import logger
from src.infrastructure.exceptions import ScoringException


class ATSScorer:
    """
    Scores a resume against a job description using ATS keyword matching logic.
    Combines exact match, fuzzy match, and section presence bonus.
    """

    SECTION_HEADERS = ["experience", "education", "skills", "summary", "projects", "work history"]
    FUZZY_THRESHOLD = 80

    def _normalize(self, text: str) -> str:
        """Normalizes text to lowercase and strips punctuation."""
        return re.sub(r"[^\w\s]", "", text.lower()).strip()

    def _extract_keywords(self, jd: JDEntity) -> list[str]:
        """Collects and deduplicates all keywords from the JD."""
        all_keywords = (
            list(jd.keywords)
            + list(jd.required_skills)
            + list(jd.preferred_skills)
        )
        # Normalize and deduplicate
        seen = set()
        unique = []
        for kw in all_keywords:
            normalized = self._normalize(kw)
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(normalized)
        return unique

    def _check_sections(self, resume_text: str) -> list[str]:
        """Checks which standard resume sections are present."""
        lower_text = resume_text.lower()
        found = [sec for sec in self.SECTION_HEADERS if sec in lower_text]
        return found

    def score(self, resume_text: str, jd: JDEntity) -> dict:
        """
        Scores a resume against a JD using ATS keyword matching.

        Returns a dict with:
          - ats_score: float (0-100)
          - matched_keywords: list[str]
          - missing_keywords: list[str]
          - keyword_match_rate: float
          - sections_found: list[str]
        """
        try:
            logger.info(f"Starting ATS scoring for JD: {jd.id}")

            keywords = self._extract_keywords(jd)
            if not keywords:
                logger.warning("No keywords found in JD — returning zero ATS score.")
                return {
                    "ats_score": 0.0,
                    "matched_keywords": [],
                    "missing_keywords": [],
                    "keyword_match_rate": 0.0,
                    "sections_found": []
                }

            normalized_resume = self._normalize(resume_text)
            resume_words = normalized_resume.split()

            matched_keywords = []
            missing_keywords = []
            total_points = 0.0

            for keyword in keywords:
                # Exact match (case insensitive, punctuation stripped)
                if keyword in normalized_resume:
                    total_points += 1.0
                    matched_keywords.append(keyword)
                    continue

                # Fuzzy match against each word in the resume
                best_ratio = max(
                    (fuzz.ratio(keyword, word) for word in resume_words),
                    default=0
                )
                if best_ratio >= self.FUZZY_THRESHOLD:
                    total_points += 0.7
                    matched_keywords.append(keyword)
                else:
                    missing_keywords.append(keyword)

            total_keywords = len(keywords)
            raw_score = (total_points / total_keywords) * 100 if total_keywords > 0 else 0.0
            keyword_match_rate = (len(matched_keywords) / total_keywords) * 100 if total_keywords > 0 else 0.0

            # Section presence bonus (max +10 points)
            sections_found = self._check_sections(resume_text)
            section_bonus = (len(sections_found) / len(self.SECTION_HEADERS)) * 10

            final_ats_score = round(min(100.0, raw_score + section_bonus), 2)

            logger.info(
                f"ATS scoring complete — score: {final_ats_score}, "
                f"matched: {len(matched_keywords)}/{total_keywords}, "
                f"sections: {sections_found}"
            )

            return {
                "ats_score": final_ats_score,
                "matched_keywords": matched_keywords,
                "missing_keywords": missing_keywords,
                "keyword_match_rate": round(keyword_match_rate, 2),
                "sections_found": sections_found,
            }

        except Exception as e:
            logger.error(f"ATS scoring failed: {str(e)}", exc_info=True)
            raise ScoringException("ATS scoring failed.", detail=str(e))
