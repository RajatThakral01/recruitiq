import re
from rapidfuzz import fuzz
from src.core.domain.entities.job_description import JDEntity
from src.infrastructure.logger import logger
from src.infrastructure.exceptions import ScoringException


# Universal stop words that apply to ALL role types
STOP_WORDS = {
    # Generic job-posting words
    'experience', 'professional', 'development',
    'engineer', 'developer', 'senior', 'junior',
    'mid', 'lead', 'manager', 'director', 'head',
    'team', 'growing', 'talented', 'looking',
    'join', 'overview', 'requirements', 'skills',
    'preferred', 'required', 'years', 'degree',
    'bachelor', 'master', 'related', 'field',
    'working', 'strong', 'knowledge', 'ability',
    'understanding', 'familiarity', 'proven',
    'excellent', 'good', 'great', 'passionate',
    'motivated', 'detail', 'oriented', 'oriented',
    'communication', 'written', 'verbal', 'oral',
    'interpersonal', 'analytical', 'creative',
    'innovative', 'dynamic', 'self', 'starter',
    'fast', 'paced', 'environment', 'role',
    'position', 'opportunity', 'company', 'org',
    'organization', 'business', 'work', 'job',
    # Generic engineering/tech concepts present on almost every resume
    'backend development', 'frontend development',
    'full stack', 'fullstack',
    'data modeling', 'problem solving', 'problem-solving',
    'algorithms', 'data structures',
    'concurrent systems', 'concurrent',
    'observability', 'scalability', 'reliability',
    'performance', 'monitoring', 'alerting', 'logging',
    'mentoring', 'debugging', 'testing',
    'deployment', 'architecture',
    'microservices', 'distributed', 'distributed systems',
    'event driven', 'cloud native',
}

# Universal synonym mapping for both tech AND non-tech common aliases.
# Multi-word phrases are applied FIRST (before word-by-word) in _normalize().
SYNONYMS = {
    # ── REST API variants (all → 'restful apis') ──────────────────
    'rest api':    'restful apis',
    'rest apis':   'restful apis',
    'restful api': 'restful apis',
    'rest':        'restful apis',
    # ── JavaScript / TypeScript ───────────────────────────────────
    'js':          'javascript',
    'ts':          'typescript',
    # ── Node.js ───────────────────────────────────────────────────
    'node':        'node.js',
    'nodejs':      'node.js',
    'node js':     'node.js',
    # ── React ─────────────────────────────────────────────────────
    'react.js':    'react',
    'reactjs':     'react',
    # ── HTML / CSS ────────────────────────────────────────────────
    'css3':        'css',
    'html5':       'html',
    # ── Databases ─────────────────────────────────────────────────
    'postgres':    'postgresql',
    'pg':          'postgresql',
    'mongo':       'mongodb',
    'mongo db':    'mongodb',
    'mysql':       'sql',
    'sqlite':      'sql',
    'nosql':       'nosql',
    'no sql':      'nosql',
    # ── Search ────────────────────────────────────────────────────
    'elastic':        'elasticsearch',
    'elastic search': 'elasticsearch',
    # ── Go ────────────────────────────────────────────────────────
    'golang':      'go',
    'go lang':     'go',
    # ── Python ────────────────────────────────────────────────────
    'py':          'python',
    # ── DevOps ────────────────────────────────────────────────────
    'k8s':         'kubernetes',
    'ci/cd':       'cicd',
    'ci cd':       'cicd',
    # ── Microsoft Office ──────────────────────────────────────────
    'ms excel':        'excel',
    'microsoft excel': 'excel',
    'ms word':         'word',
    # ── Analytics / Marketing ─────────────────────────────────────
    'google analytics':       'analytics',
    'social media marketing': 'social media',
    'search engine optimization': 'seo',
    'search engine marketing':    'sem',
    'pay per click':              'ppc',
    'customer relationship':      'crm',
    # ── AI / ML ───────────────────────────────────────────────────
    'machine learning':    'ml',
    'artificial intelligence': 'ai',
    # ── UX / UI ───────────────────────────────────────────────────
    'user experience': 'ux',
    'user interface':  'ui',
    # ── Management ────────────────────────────────────────────────
    'project management': 'pm',
    'product management': 'pm',
    # ── Business metrics ──────────────────────────────────────────
    'key performance':      'kpi',
    'return on investment': 'roi',
}


class ATSScorer:
    """
    Scores a resume against a job description using industry-standard
    multi-factor ATS logic.

    Scoring breakdown:
      Factor 1 – Hard Skills Match       40 pts max
      Factor 2 – Experience Alignment    30 pts max
      Factor 3 – Education / Certs       15 pts max
      Factor 4 – Context / NLP Quality   15 pts max
      ─────────────────────────────────────────────
      Total                             100 pts max
    """

    SECTION_HEADERS = ["experience", "education", "skills", "summary", "projects", "work history"]
    FUZZY_THRESHOLD = 80

    # ── Factor 2 seniority signals ────────────────────────────────────────
    SENIOR_SIGNALS = [
        'led', 'lead', 'senior', 'sr.', 'principal',
        'staff', 'architect', 'head of', 'director',
        'manager', 'managed', 'mentored', 'owned',
        'spearheaded', 'founded', 'built and scaled',
    ]
    JUNIOR_SIGNALS = [
        'intern', 'internship', 'junior', 'jr.',
        'trainee', 'fresher', 'entry level', 'graduate',
    ]

    # ── Factor 3 certification keywords ──────────────────────────────────
    CERT_KEYWORDS = [
        'certified', 'certification', 'certificate',
        'aws certified', 'google certified', 'pmp',
        'cissp', 'cpa', 'cfa', 'rhcsa', 'comptia',
        'azure certified', 'gcp certified', 'scrum',
        'agile certified', 'six sigma',
    ]

    # ── Factor 4 NLP quality signals ──────────────────────────────────────
    ACTION_VERBS = [
        'built', 'developed', 'designed', 'implemented',
        'led', 'managed', 'created', 'launched', 'scaled',
        'improved', 'reduced', 'increased', 'delivered',
        'optimized', 'automated', 'architected', 'deployed',
        'migrated', 'integrated', 'established', 'drove',
        'owned', 'mentored', 'spearheaded', 'transformed',
    ]
    # Matches: 40%, 200K users, 5+ requests, $1M, 3 hours, 200ms, 3x faster, 10 pts
    # Unified pattern – kept in sync with SemanticScorer.METRIC_PATTERN
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

    def _normalize(self, text: str) -> str:
        """Normalizes text to lowercase, strips most punctuation, and applies synonym mapping.

        Processing order (important — prevents double-expansion):
          1. Lowercase + strip
          2. Remove characters that are not word chars, spaces, '.', '+', '#'
          3. Replace multi-word phrases first (e.g. 'rest api' → 'restful apis')
          4. Word-by-word synonym replacement (e.g. 'nodejs' → 'node.js')

        Safe to call on both individual keywords AND full resume text.
        """
        text = text.lower().strip()
        text = re.sub(r'[^\w\s\.\+\#]', ' ', text)

        # Step 3 – multi-word phrase substitution (longest-match wins naturally
        # because we replace left-to-right and break on first hit per position)
        for phrase, replacement in SYNONYMS.items():
            if ' ' in phrase:
                text = text.replace(phrase, replacement)

        # Step 4 – word-by-word substitution for remaining tokens
        words = text.split()
        normalized = [SYNONYMS.get(w, w) for w in words]
        return ' '.join(normalized)

    def _extract_keywords(self, jd: JDEntity) -> list[str]:
        """Collects and deduplicates all keywords from the JD, filtering stop words."""
        all_keywords = (
            list(jd.keywords)
            + list(jd.required_skills)
            + list(jd.preferred_skills)
        )
        # Normalize, filter stop words, and deduplicate
        seen = set()
        unique = []
        for kw in all_keywords:
            normalized = self._normalize(kw)
            if normalized and normalized not in seen and normalized not in STOP_WORDS and len(normalized) > 2:
                seen.add(normalized)
                unique.append(normalized)
        return unique

    def _check_sections(self, resume_text: str) -> list[str]:
        """Checks which standard resume sections are present."""
        lower_text = resume_text.lower()
        found = [sec for sec in self.SECTION_HEADERS if sec in lower_text]
        return found

    def _extract_experience_section(self, resume_text: str) -> str:
        """Extracts text between the experience header and the next major section header.
        Returns an empty string if no experience section is found.
        """
        lower = resume_text.lower()
        exp_headers = ['work experience', 'professional experience', 'experience', 'work history', 'employment']
        start_idx = -1
        for header in exp_headers:
            idx = lower.find(header)
            if idx != -1:
                start_idx = idx + len(header)
                break
        if start_idx == -1:
            return ''
        # Find the next major section header after the experience block
        next_section = re.search(
            r'\n(?:education|skills|projects|summary|certifications|awards|references|publications)\b',
            resume_text[start_idx:],
            re.IGNORECASE,
        )
        if next_section:
            return resume_text[start_idx: start_idx + next_section.start()]
        return resume_text[start_idx:]

    def _filter_and_normalize_keywords(self, keywords: list[str]) -> list[str]:
        """Helper to normalize and filter a list of keywords."""
        seen = set()
        unique = []
        for kw in keywords:
            normalized = self._normalize(kw)
            if normalized and normalized not in seen and normalized not in STOP_WORDS and len(normalized) > 2:
                seen.add(normalized)
                unique.append(normalized)
        return unique

    def _match_keywords(self, keywords: list[str], normalized_resume: str, resume_words: list[str]) -> tuple[list[str], list[str]]:
        """Match keywords against resume text. Returns (matched, missing)."""
        matched = []
        missing = []
        for keyword in keywords:
            # Exact match (case insensitive, punctuation stripped)
            if keyword in normalized_resume:
                matched.append(keyword)
                continue
            # Fuzzy match against each word in the resume
            best_ratio = max(
                (fuzz.ratio(keyword, word) for word in resume_words),
                default=0
            )
            if best_ratio >= self.FUZZY_THRESHOLD:
                matched.append(keyword)
            else:
                missing.append(keyword)
        return matched, missing

    def score(self, resume_text: str, jd: JDEntity) -> dict:
        """
        Industry-standard multi-factor ATS score.

        Factor 1 – Hard Skills Match    (max 40 pts):
          base = required*0.70 + preferred*0.30, scaled to 40;
          + frequency bonus (keyword repetition, max +10);
          + context bonus (required keywords in experience section, max +10).

        Factor 2 – Experience Alignment (max 30 pts):
          Senior/junior signal counts vs JD seniority level.

        Factor 3 – Education / Certs    (max 15 pts):
          Certification keyword presence.

        Factor 4 – Context / NLP Quality (max 15 pts):
          Action verb count + quantitative metric count.

        Returns a dict with:
          - ats_score: float (0-100)
          - matched_keywords: list[str]
          - missing_keywords: list[str]
          - keyword_match_rate: float
          - sections_found: list[str]
        """
        try:
            logger.info(f"Starting ATS scoring for JD: {jd.id}")

            # ── keyword prep ─────────────────────────────────────────────
            must_have_keywords = self._filter_and_normalize_keywords(jd.must_have)
            required_keywords  = self._filter_and_normalize_keywords(jd.required_skills)
            preferred_keywords = self._filter_and_normalize_keywords(jd.preferred_skills)
            existing  = set(must_have_keywords + required_keywords + preferred_keywords)
            other_raw = [k for k in jd.keywords if self._normalize(k) not in existing]
            other_keywords = self._filter_and_normalize_keywords(other_raw)

            # Apply full synonym normalization to resume text before matching
            normalized_resume = self._normalize(resume_text)
            resume_words = normalized_resume.split()

            matched_must_have, missing_must_have = self._match_keywords(
                must_have_keywords, normalized_resume, resume_words)
            matched_required,  missing_required  = self._match_keywords(
                required_keywords,  normalized_resume, resume_words)
            matched_preferred, missing_preferred = self._match_keywords(
                preferred_keywords, normalized_resume, resume_words)
            matched_other, _                     = self._match_keywords(
                other_keywords,     normalized_resume, resume_words)

            sections_found = self._check_sections(resume_text)

            # ── FACTOR 1: Hard Skills Match (max 40 pts) ─────────────────
            must_have_score = (len(matched_must_have) / len(must_have_keywords)) * 100 \
                              if must_have_keywords else 50.0
            required_score  = (len(matched_required)  / len(required_keywords))  * 100 \
                              if required_keywords else 50.0
            preferred_score = (len(matched_preferred) / len(preferred_keywords)) * 100 \
                              if preferred_keywords else 50.0
            # Prioritize must-have > required > preferred skills.
            base_skills_score = (
                must_have_score * 0.50
                + required_score * 0.35
                + preferred_score * 0.15
            )

            # Frequency bonus: how many times each matched keyword appears (max +10)
            lower_resume = resume_text.lower()
            freq_bonus = 0.0
            for kw in matched_must_have + matched_required + matched_preferred:
                occurrences = lower_resume.count(kw.lower())
                if occurrences >= 4:
                    freq_bonus += 3
                elif occurrences >= 2:
                    freq_bonus += 2
                # 1 occurrence: no bonus
            freq_bonus = min(10.0, freq_bonus)

            # Context bonus: required keywords present in experience section (max +10)
            experience_text = self._extract_experience_section(resume_text)
            context_bonus = 0.0
            if experience_text:
                lower_exp = experience_text.lower()
                for kw in matched_must_have + matched_required:
                    if kw.lower() in lower_exp:
                        context_bonus += 3
            context_bonus = min(10.0, context_bonus)

            hard_skills_score = min(40.0, base_skills_score * 0.20 + freq_bonus + context_bonus)

            # ── FACTOR 2: Experience Alignment (max 30 pts) ──────────────
            lower_text   = resume_text.lower()
            senior_count = sum(1 for s in self.SENIOR_SIGNALS if s in lower_text)
            junior_count = sum(1 for s in self.JUNIOR_SIGNALS if s in lower_text)

            jd_level = (jd.seniority_level or "").lower()
            if jd_level in ("senior", "lead"):
                if senior_count >= 3:
                    seniority_score = 30
                elif senior_count >= 1:
                    seniority_score = 20
                elif junior_count > senior_count:
                    seniority_score = 10
                else:
                    seniority_score = 15
            elif jd_level in ("junior", "mid"):
                if junior_count >= 1 or senior_count >= 1:
                    seniority_score = 25
                else:
                    seniority_score = 20
            else:
                seniority_score = 20  # neutral default

            experience_alignment_score = float(seniority_score)

            # ── FACTOR 3: Education / Certification Match (max 15 pts) ───
            cert_count = sum(1 for c in self.CERT_KEYWORDS if c in lower_text)
            if cert_count >= 2:
                cert_bonus = 15
            elif cert_count == 1:
                cert_bonus = 10
            else:
                cert_bonus = 5
            education_score = float(cert_bonus)

            # ── FACTOR 4: Context / NLP Quality (max 15 pts) ─────────────
            action_verb_count = sum(1 for v in self.ACTION_VERBS if v in lower_text)
            metric_count      = len(self.METRIC_PATTERN.findall(resume_text))

            if   action_verb_count >= 8 and metric_count >= 3:
                nlp_score = 15
            elif action_verb_count >= 5 and metric_count >= 2:
                nlp_score = 12
            elif action_verb_count >= 3 and metric_count >= 1:
                nlp_score = 8
            elif action_verb_count >= 2:
                nlp_score = 5
            else:
                nlp_score = 2
            context_nlp_score = float(nlp_score)

            # ── Domain relevance multiplier for Factors 2, 3, 4 ────────────
            # Ratio of JD required skills found anywhere in the resume.
            # Candidates with no domain overlap get a floor reduction;
            # strong matches are unpenalised.
            raw_required = self._filter_and_normalize_keywords(jd.required_skills)
            if raw_required:
                domain_matched = sum(
                    1 for kw in raw_required if kw in normalized_resume
                )
                domain_relevance_ratio = domain_matched / len(raw_required)
            else:
                domain_matched = 0
                domain_relevance_ratio = 1.0  # no required skills → no penalty

            logger.info(
                f"Domain relevance ratio: {domain_relevance_ratio:.2f} "
                f"({domain_matched}/{len(raw_required)} required skills matched)"
            )

            # Apply multiplier with per-factor floors
            experience_alignment_score = round(
                experience_alignment_score * max(0.3, domain_relevance_ratio)
            )
            education_score = round(
                education_score * max(0.5, domain_relevance_ratio)
            )
            context_nlp_score = round(
                context_nlp_score * max(0.4, domain_relevance_ratio)
            )

            # ── Final ATS score ───────────────────────────────────────────
            keyword_boost = min(20.0, base_skills_score * 0.20)
            final_ats_score = round(
                min(100.0,
                    hard_skills_score
                    + keyword_boost
                    + experience_alignment_score
                    + education_score
                    + context_nlp_score),
                2,
            )

            # ── Reporting ─────────────────────────────────────────────────
            all_matched_keywords = matched_must_have + matched_required + matched_preferred + matched_other
            missing_keywords     = missing_must_have + missing_required + missing_preferred
            total_scored         = len(must_have_keywords) + len(required_keywords) + len(preferred_keywords)
            keyword_match_rate   = (
                (len(matched_must_have) + len(matched_required) + len(matched_preferred)) / total_scored * 100
                if total_scored > 0 else 0.0
            )

            logger.info(
                f"ATS breakdown — "
                f"skills:{hard_skills_score:.1f} "
                f"exp:{experience_alignment_score:.1f} (domain_ratio={domain_relevance_ratio:.2f}) "
                f"edu:{education_score:.1f} "
                f"nlp:{context_nlp_score:.1f} "
                f"total:{final_ats_score:.1f}"
            )

            return {
                "ats_score":         final_ats_score,
                "matched_keywords":  all_matched_keywords,
                "missing_keywords":  missing_keywords,
                "keyword_match_rate": round(keyword_match_rate, 2),
                "sections_found":    sections_found,
            }

        except Exception as e:
            logger.error(f"ATS scoring failed: {str(e)}", exc_info=True)
            raise ScoringException("ATS scoring failed.", detail=str(e))
