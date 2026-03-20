import re
from rapidfuzz import fuzz
from src.core.domain.entities.job_description import JDEntity
from src.core.ports.embedding_port import EmbeddingPort
from src.infrastructure.logger import logger
from src.infrastructure.exceptions import ScoringException


IMPACT_WORDS = [
    'built', 'scaled', 'improved', 'reduced',
    'increased', 'led', 'designed', 'developed',
    'launched', 'optimized', 'automated',
    'managed', 'created', 'delivered', 'grew',
    'achieved', 'generated', 'saved', 'drove',
    'executed', 'implemented', 'established',
    'transformed', 'negotiated', 'secured',
    'expanded', 'trained', 'mentored', 'owned',
    'spearheaded', 'oversaw', 'coordinated',
    'streamlined', 'revamped', 'pioneered'
]

class SemanticScorer:
    """
    Scores resume sections (skills, projects) using sentence-transformers embeddings
    and heuristic project quality signals.
    """

    # Unified metric regex shared by score_projects() and any future methods.
    # Matches: 40%, 200K users, 5+ requests, $1M, 3 hours, 200ms, 3x faster, 10 pts
    METRIC_PATTERN = re.compile(
        r'(\d+%'
        r'|\d+[kmKM]?\+?\s*(?:users?|requests?|services?|customers?)'
        r'|\$\s*\d+'
        r'|\d+\s*(?:hours?|days?|weeks?|months?)'
        r'|\d+\s?ms'
        r'|\d+x\s?(?:faster|improvement|growth|increase)'
        r'|\d+\s?(?:pts?|points?)'
        r')',
        re.IGNORECASE,
    )

    # Signals that indicate genuine project depth / technical sophistication
    DEPTH_SIGNALS = [
        'architecture', 'distributed', 'scalable',
        'production', 'real-time', 'microservice',
        'algorithm', 'optimization', 'performance',
        'security', 'api', 'database', 'pipeline',
        'deployed', 'open source', 'patent',
        'published', 'research', 'machine learning',
        'system design', 'infrastructure',
    ]

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
        a combined resume text built from three sources:
          1. Skills section (if found)
          2. First 1500 chars of the experience/work section
          3. First 500 chars of the projects section

        Returns a float score 0-100.
        """
        try:
            logger.info(f"Semantic skills scoring for JD: {jd.id}")

            # Build JD comparison text from all skill-related fields
            jd_skills_text = " ".join(
                jd.required_skills +
                jd.preferred_skills +
                jd.keywords
            )

            # 1. Skills section
            skills_text = self._extract_section(
                resume_text,
                "skills", "technical skills", "core skills",
                "key skills", "competencies", "expertise",
                "technologies", "tech stack", "capabilities",
                "qualifications", "areas of expertise",
                "professional skills", "tools",
                fallback_chars=2000
            )

            # 2. Experience / work section (up to 1500 chars)
            experience_text = self._extract_section(
                resume_text,
                "experience", "work experience", "employment",
                "work history", "professional experience",
                fallback_chars=0
            )[:1500]

            # 3. Projects section (up to 500 chars)
            projects_text = self._extract_section(
                resume_text,
                "projects", "open source",
                fallback_chars=0
            )[:500]

            # Combine all three sources
            resume_comparison_text = " ".join(
                part for part in (skills_text, experience_text, projects_text) if part.strip()
            )

            if not jd_skills_text.strip() or not resume_comparison_text.strip():
                return 0.0

            similarity = self.embedding_port.get_similarity(jd_skills_text, resume_comparison_text)

            # Arcee Adapter returns float 0-1 (already clamped)
            # scale to 0-100
            score = round(similarity * 100, 2)

            # Domain relevance adjustment
            resume_lower = resume_comparison_text.lower()

            # Count how many required skills appear in resume
            required_words = [w.strip() for w in jd.required_skills if len(w.strip()) > 2]

            if required_words:
                matched_count = sum(
                    1 for skill in required_words
                    if skill.lower() in resume_lower
                )
                domain_ratio = matched_count / len(required_words)
                # More nuanced multiplier:
                # High raw similarity (≥60) means semantic match is already strong
                # → apply a gentle floor so equivalent-tech candidates aren't penalised.
                # Low raw similarity means skills are genuinely different
                # → apply a stricter floor.
                raw_score_before_multiplier = score

                if raw_score_before_multiplier >= 60:
                    multiplier = max(0.75, domain_ratio)
                elif raw_score_before_multiplier >= 40:
                    multiplier = max(0.55, domain_ratio)
                else:
                    multiplier = max(0.35, domain_ratio)

                score = round(raw_score_before_multiplier * multiplier, 2)
                logger.info(
                    f"Skills domain adjustment: "
                    f"raw={raw_score_before_multiplier} "
                    f"{matched_count}/{len(required_words)} "
                    f"required skills found, "
                    f"domain_ratio={domain_ratio:.2f} "
                    f"multiplier={multiplier:.2f}, "
                    f"adjusted_score={score}"
                )

            logger.info(
                f"Semantic skills score: {score} "
                f"(skills_chars={len(skills_text)}, "
                f"exp_chars={len(experience_text)}, "
                f"proj_chars={len(projects_text)})"
            )
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
            if impact_words_found >= 8:
                impact_score = 40.0
            elif impact_words_found >= 5:
                impact_score = 30.0
            elif impact_words_found >= 3:
                impact_score = 20.0
            elif impact_words_found >= 1:
                impact_score = 10.0
            else:
                impact_score = 0.0

            # Metrics
            metrics_found = len(self.METRIC_PATTERN.findall(resume_text))
            if metrics_found >= 5:
                metric_score = 30.0
            elif metrics_found >= 3:
                metric_score = 22.0
            elif metrics_found >= 2:
                metric_score = 15.0
            elif metrics_found >= 1:
                metric_score = 8.0
            else:
                metric_score = 0.0

            # Tech relevance: fuzzy match both required_skills AND keywords
            # against the project/experience text (deduplicated)
            combined_tech_terms = list({
                t.lower()
                for t in (jd.required_skills + jd.keywords)
                if t.strip()
            })
            if combined_tech_terms:
                matching_skills = sum(
                    1
                    for term in combined_tech_terms
                    if fuzz.partial_ratio(term, project_text) >= 75
                )
                tech_score = min(
                    30.0,
                    (matching_skills / max(len(jd.required_skills), 1)) * 30
                )
            else:
                tech_score = 0.0

            # Depth bonus: rewards technically sophisticated project descriptions
            depth_count = sum(1 for s in self.DEPTH_SIGNALS if s in project_text)
            if depth_count >= 5:
                depth_bonus = 10
            elif depth_count >= 3:
                depth_bonus = 6
            elif depth_count >= 1:
                depth_bonus = 3
            else:
                depth_bonus = 0

            total = min(100, round(
                impact_score + metric_score + tech_score + depth_bonus, 2
            ))

            logger.info(
                f"Project scoring — "
                f"impact:{impact_score} "
                f"metrics:{metric_score} "
                f"tech:{tech_score} "
                f"depth:{depth_bonus} "
                f"total:{total}"
            )
            return total

        except Exception as e:
            logger.error(f"Project scoring failed: {str(e)}", exc_info=True)
            raise ScoringException("Project scoring failed.", detail=str(e))
