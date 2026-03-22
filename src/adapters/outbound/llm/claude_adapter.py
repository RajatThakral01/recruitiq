import json
import re
import time
from typing import Any, Callable, Dict

import openai
from src.core.ports.llm_port import LLMPort
from src.adapters.outbound.llm.prompt_templates import (
  build_experience_prompt,
  build_projects_prompt,
  build_skills_prompt,
)
from src.services.llm_response_parser import LLMResponseParser
from src.services.llm_context_builder import build_llm_scoring_context
from src.infrastructure.config import settings
from src.infrastructure.logger import logger, log_event
from src.infrastructure.exceptions import (
  LLMAuthenticationException,
  LLMTimeoutException,
  ScoringException,
)


class BaseLLMAdapter(LLMPort):
  """Shared OpenAI-compatible LLM implementation used by outbound adapters."""

  def __init__(self) -> None:
    # This base init is intentionally left minimal.
    # Subclasses (GrokAdapter) handle client/model/key setup themselves.
    self._auth_failed = False
    logger.info("BaseLLMAdapter initialized (subclass handles client setup).")

  def _safe_score(self, val: Any, default: float = 50.0) -> float:
    """Safely converts an LLM-provided score to a 0-100 float.

    If the value is missing or not a valid number, returns the provided default.
    """
    return LLMResponseParser.as_score(val, default=default)

  def _strip_markdown(self, text: str) -> str:
    """Strip markdown code fences from LLM response."""
    return LLMResponseParser.strip_markdown(text)

  def _extract_json_candidate(self, text: str) -> str:
    """Extract probable JSON object payload from an LLM response string."""
    return LLMResponseParser.extract_json_object(text)

  def _as_string(self, value: Any, default: str = "") -> str:
    return LLMResponseParser.as_string(value, default=default)

  def _as_string_list(self, value: Any) -> list[str]:
    return LLMResponseParser.as_string_list(value)

  def _normalize_project_items(self, value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
      return []
    normalized: list[dict[str, Any]] = []
    for item in value:
      if not isinstance(item, dict):
        continue
      normalized.append(
        {
          "name": self._as_string(item.get("name", "")),
          "description": self._as_string(item.get("description", "")),
          "tech_stack": self._as_string_list(item.get("tech_stack", [])),
          "metrics": self._as_string(item.get("metrics", "")),
        }
      )
    return normalized

  def _validate_resume_schema(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
      "candidate_name": self._as_string(payload.get("candidate_name", "")),
      "email": self._as_string(payload.get("email", "")) or None,
      "years_experience": max(0.0, float(payload.get("years_experience", 0.0) or 0.0)),
      "relevant_experience": max(0.0, float(payload.get("relevant_experience", 0.0) or 0.0)),
      "education_level": self._as_string(payload.get("education_level", "Other")) or "Other",
      "parsed_skills": self._as_string_list(payload.get("parsed_skills", [])),
      "projects": self._normalize_project_items(payload.get("projects", [])),
    }

  def _validate_jd_schema(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
      "title": self._as_string(payload.get("title", "")),
      "seniority_level": self._as_string(payload.get("seniority_level", "junior")) or "junior",
      "min_experience": max(0.0, float(payload.get("min_experience", 0.0) or 0.0)),
      "education_requirement": self._as_string(payload.get("education_requirement", "Other")) or "Other",
      "required_skills": self._as_string_list(payload.get("required_skills", [])),
      "preferred_skills": self._as_string_list(payload.get("preferred_skills", [])),
      "must_have": self._as_string_list(payload.get("must_have", [])),
      "nice_to_have": self._as_string_list(payload.get("nice_to_have", [])),
      "keywords": self._as_string_list(payload.get("keywords", [])),
    }

  def _validate_skills_schema(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
      "skills_score": self._safe_score(payload.get("skills_score", 50.0), default=50.0),
      "direct_matches": self._as_string_list(payload.get("direct_matches", [])),
      "transferable_matches": self._as_string_list(payload.get("transferable_matches", [])),
      "missing_critical": self._as_string_list(payload.get("missing_critical", [])),
      "reasoning": self._as_string(payload.get("reasoning", "")),
    }

  def _validate_projects_schema(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
      "project_score": self._safe_score(payload.get("project_score", 40.0), default=40.0),
      "impact_score": self._safe_score(payload.get("impact_score", 40.0), default=40.0),
      "complexity_score": self._safe_score(payload.get("complexity_score", 40.0), default=40.0),
      "relevance_score": self._safe_score(payload.get("relevance_score", 40.0), default=40.0),
      "ownership_score": self._safe_score(payload.get("ownership_score", 40.0), default=40.0),
      "best_project": self._as_string(payload.get("best_project", "")),
      "reasoning": self._as_string(payload.get("reasoning", "")),
    }

  def _validate_experience_schema(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
      "experience_score": self._safe_score(payload.get("experience_score", 40.0), default=40.0),
      "domain_match_score": self._safe_score(payload.get("domain_match_score", 40.0), default=40.0),
      "seniority_match_score": self._safe_score(payload.get("seniority_match_score", 40.0), default=40.0),
      "trajectory_score": self._safe_score(payload.get("trajectory_score", 40.0), default=40.0),
      "relevant_years": max(0.0, float(payload.get("relevant_years", 0.0) or 0.0)),
      "employment_type": self._as_string(payload.get("employment_type", "unknown")) or "unknown",
      "reasoning": self._as_string(payload.get("reasoning", "")),
    }

  def _validate_strengths_gaps_schema(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
      "strengths": self._as_string_list(payload.get("strengths", [])),
      "gaps": self._as_string_list(payload.get("gaps", [])),
    }

  def _extract_retry_after_seconds(self, error: Exception) -> float:
    """Best-effort parse of provider retry hint from error messages.

    Groq errors often contain strings like: "Please try again in 470ms" or "10.435s".
    """
    message = str(error)
    # Format like: "Please try again in 12m45.504s"
    ms_combo = re.search(
      r"try again in\s+([0-9]+)m([0-9]+(?:\.[0-9]+)?)s",
      message,
      re.IGNORECASE,
    )
    if ms_combo:
      minutes = float(ms_combo.group(1))
      seconds = float(ms_combo.group(2))
      return max(0.0, minutes * 60.0 + seconds)

    ms_match = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)ms", message, re.IGNORECASE)
    if ms_match:
      return max(0.0, float(ms_match.group(1)) / 1000.0)

    s_match = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", message, re.IGNORECASE)
    if s_match:
      return max(0.0, float(s_match.group(1)))

    return 0.0

  def _call_llm_json(
    self,
    *,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    label: str,
    validator: Callable[[Dict[str, Any]], Dict[str, Any]],
    default_payload: Dict[str, Any],
    parse_retries: int = 2,
  ) -> Dict[str, Any]:
    """Call LLM, parse JSON robustly, validate schema, and fallback safely."""
    operation_start = time.perf_counter()
    for parse_attempt in range(1, parse_retries + 1):
      try:
        raw = self._call_llm(system_prompt, user_content, max_tokens=max_tokens)
        parsed = LLMResponseParser.parse_json_object(raw)
        validated = validator(parsed)
        log_event(
          "info",
          "llm_json_validation_succeeded",
          label=label,
          parse_attempt=parse_attempt,
          latency_seconds=f"{time.perf_counter() - operation_start:.3f}",
        )
        return validated
      except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning(
          f"{label}: JSON validation failed on attempt {parse_attempt}/{parse_retries}: {e}"
        )
        log_event(
          "warning",
          "llm_parse_failure",
          label=label,
          parse_attempt=parse_attempt,
          parse_retries=parse_retries,
          error_type=type(e).__name__,
        )
      except Exception as e:
        if isinstance(e, LLMAuthenticationException):
          logger.error(f"{label}: terminal LLM auth failure: {e}")
          log_event(
            "error",
            "llm_call_terminal_auth_failure",
            label=label,
            parse_attempt=parse_attempt,
            parse_retries=parse_retries,
            error_type=type(e).__name__,
          )
          raise
        if isinstance(e, LLMTimeoutException):
          # Avoid duplicate parse-level retries when call-level retries are already exhausted.
          logger.error(f"{label}: terminal LLM timeout: {e}")
          log_event(
            "error",
            "llm_call_terminal_timeout",
            label=label,
            parse_attempt=parse_attempt,
            parse_retries=parse_retries,
            error_type=type(e).__name__,
          )
          break
        logger.error(f"{label}: LLM call failed on attempt {parse_attempt}/{parse_retries}: {e}")
        log_event(
          "error",
          "llm_call_or_validation_error",
          label=label,
          parse_attempt=parse_attempt,
          parse_retries=parse_retries,
          error_type=type(e).__name__,
        )

    logger.warning(f"{label}: returning fallback payload after retry exhaustion")
    log_event(
      "warning",
      "llm_fallback_payload_used",
      label=label,
      parse_retries=parse_retries,
      latency_seconds=f"{time.perf_counter() - operation_start:.3f}",
    )
    return default_payload

  def _call_llm(self, system_prompt: str, user_content: str, max_tokens: int) -> str:
    """Helper to call configured LLM with exponential backoff retry.
    Retry/timeout/backoff are controlled via settings.
    """
    max_retries = max(1, int(settings.LLM_MAX_RETRIES or 1))
    timeout_seconds = float(settings.LLM_TIMEOUT_SECONDS or 45.0)
    backoff_base = max(0.0, float(settings.LLM_BACKOFF_BASE_SECONDS or 0.0))

    if getattr(self, "_auth_failed", False):
      raise LLMAuthenticationException(
        "LLM authentication previously failed; skipping further requests.",
        detail="Verify MISTRAL_API_KEY and restart the API service.",
      )

    for attempt in range(1, max_retries + 1):
      call_start = time.perf_counter()
      try:
        logger.info(
          f"LLM call attempt {attempt}/{max_retries} "
          f"model={self.model} timeout={timeout_seconds}s"
        )
        response = self.client.chat.completions.create(
          model=self.model,
          messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
          ],
          max_tokens=max_tokens,
          temperature=0.0,
          timeout=timeout_seconds,
        )
        content = response.choices[0].message.content
        if not content:
          raise ScoringException("Received empty response from LLM provider.")
        log_event(
          "info",
          "llm_call_succeeded",
          model=self.model,
          attempt=attempt,
          max_retries=max_retries,
          latency_seconds=f"{time.perf_counter() - call_start:.3f}",
        )
        return content.strip()
      except Exception as e:
        error_text = str(e).lower()
        if (
          "authenticationerror" in type(e).__name__.lower()
          or "unauthorized" in error_text
          or "error code: 401" in error_text
        ):
          setattr(self, "_auth_failed", True)
          raise LLMAuthenticationException(
            "Mistral authentication failed (401 Unauthorized).",
            detail="Set a valid MISTRAL_API_KEY and restart the API service.",
          )

        logger.error(f"LLM call failed on attempt {attempt}: {e}")
        log_event(
          "warning",
          "llm_call_retry",
          model=self.model,
          attempt=attempt,
          max_retries=max_retries,
          error_type=type(e).__name__,
          latency_seconds=f"{time.perf_counter() - call_start:.3f}",
        )
        if attempt == max_retries:
          raise LLMTimeoutException(
            f"LLM API call failed after {max_retries} retries.",
            detail=str(e),
          )

        # Respect provider retry hints when present to reduce repeated 429s.
        hinted_sleep = self._extract_retry_after_seconds(e)
        long_cooldown_seconds = 60.0
        if "tokens per day" in error_text and hinted_sleep >= long_cooldown_seconds:
          raise LLMTimeoutException(
            "LLM rate limit cooldown too long for synchronous request.",
            detail=f"retry_after_seconds={hinted_sleep:.3f}; error={e}",
          )

        exponential_sleep = backoff_base * (2 ** (attempt - 1))
        sleep_seconds = max(hinted_sleep, exponential_sleep)

        if sleep_seconds > 0:
          logger.info(
            f"LLM retry backoff sleep={sleep_seconds:.3f}s "
            f"(hinted={hinted_sleep:.3f}s, exponential={exponential_sleep:.3f}s)"
          )
          time.sleep(sleep_seconds)
    raise LLMTimeoutException("LLM API call exhausted retries.")

  def parse_resume(self, text: str) -> Dict[str, Any]:
        system_prompt = """
You are an expert resume parser.
You work with ALL types of resumes - technical,
non-technical, creative, managerial, and more.

Extract structured data and return ONLY valid JSON:
{
  "candidate_name": "Full Name",
  "email": "email or null",
  "years_experience": <calculate total months of ALL professional experience
    including internships, part-time roles, research positions, and contract work,
    then convert to years as a float. Add up ALL roles listed, include internships,
    for current roles calculate until today.
    Convert total months to years rounded to 1 decimal.
    IMPORTANT: years_experience is the TOTAL duration of employment, NOT skill-specific experience.
    Do NOT attribute the full role duration to every technology mentioned in that role.
    A person who used PostgreSQL for 2 months in a 1-year role has 2 months of PostgreSQL experience, not 1 year.>,

  "relevant_experience": <float - ONLY count experience where the candidate's
    PRIMARY DAILY WORK directly matches the job type.

    Be EXTREMELY STRICT and CONSERVATIVE. When in doubt, return 0.0.
    - Backend engineer role: ONLY count months where candidate wrote backend
      code daily as their PRIMARY job. Data analysis, growth, marketing,
      operations, research = 0.0. No exceptions.
    - Data analyst role: ONLY count months doing actual data analysis daily.
      Backend, ops, marketing, growth = 0.0.
    - AI/ML engineer role: ONLY count months building AI/ML systems daily.
      Data analysis without model building = 0.0.
    - Marketing role: ONLY count marketing work. Engineering, finance = 0.0.
    - If a candidate has ONLY done internships in adjacent or different domains,
      relevant_experience should be 0.0 or at most 0.2.
    - A data analyst applying for a backend role has 0.0 relevant experience.
    - A growth intern applying for an engineering role has 0.0 relevant experience.
    - NEVER give credit for transferable skills here — that is handled elsewhere.
      This field is ONLY about direct domain match of daily work.

    Multipliers by role type:
    - Full-time role in same domain: count 100% (1.0x)
    - Internship in same domain: count 50% (0.5x multiplier on months)
    - Research in same domain: count 30% (0.3x multiplier on months)
    - Work in a completely different domain: count 0.0

    Concrete examples:
    For a data analyst applying to a backend engineer role:
      Data analyst work = 0.0 relevant years (different domain)
      Growth intern work = 0.0 relevant years (different domain)
      Total relevant_experience = 0.0

    For a backend engineer applying to a backend engineer role:
      Full-time SDE II (1 year) = 1.0 year relevant
      Backend intern (6 months) = 0.5 * 0.5x = 0.25 years relevant
      Frontend intern (3 months) = 0.0 (different domain)
      Total relevant_experience = 1.25 years

    For a full-stack intern applying to a backend role:
      Fullstack intern (6 months) = 0.5 * 0.5x = 0.25 years relevant
      Research intern = 0.0 (different domain)
      Total relevant_experience = 0.25 years

    When in doubt, be conservative (round down).
    Return as float rounded to 1 decimal place.>,
  "education_level": "Bachelor|Master|PhD|Other",
  "parsed_skills": ["skill1", "skill2"],
  "projects": [
    {
      "name": "project or achievement name",
      "description": "what was done",
      "tech_stack": ["tool1", "tool2"],
      "metrics": "measurable impact or null"
    }
  ]
}

Rules for ALL fields:
- parsed_skills: extract EVERY skill, tool, 
  software, methodology, competency mentioned 
  ANYWHERE in the resume. Be comprehensive.
  For tech roles: React, Python, AWS etc.
  For non-tech roles: Salesforce, Excel, 
  copywriting, budget management, SEO etc.
  Include ALL of them, minimum 5 skills.


- years_experience: total months of ALL professional
  experience (including internships, part-time, research,
  contract). Add up ALL roles. For current roles calculate
  until today. Never use only the latest role duration.
  Convert total months to years, round to 1 decimal.
  This is TOTAL employment duration only — not skill duration.
  Do NOT say someone has X years of a specific technology
  just because they worked at a company for X years.
  Only count months where that specific skill was actively used daily.

- relevant_experience: STRICT domain match only.
  Full-time same domain = 100% credit.
  Internship same domain = 50% credit (0.5x months).
  Research same domain = 30% credit (0.3x months).
  Different domain entirely = 0.0 credit.
  Aggregate credit across all relevant roles, not only latest role.
  When in doubt, be conservative. Round to 1 decimal.

- projects: look in BOTH the PROJECTS section 
  AND the EXPERIENCE section for achievements.
  For non-tech roles, "projects" means campaigns 
  run, initiatives led, processes improved etc.
  Include up to 5 most impactful ones.

- metrics: look for numbers, percentages, 
  dollar amounts, user counts, time savings,
  revenue impact, team sizes etc.
  Examples: "increased revenue by 40%",
  "managed $2M budget", "grew team from 3 to 12"

Return only JSON. No explanation. No markdown.
"""
        return self._call_llm_json(
          system_prompt=system_prompt,
          user_content=text,
          max_tokens=1000,
          label="parse_resume",
          validator=self._validate_resume_schema,
          default_payload={
            "candidate_name": "",
            "email": None,
            "years_experience": 0.0,
            "relevant_experience": 0.0,
            "education_level": "Other",
            "parsed_skills": [],
            "projects": [],
          },
        )

  def parse_jd(self, text: str) -> Dict[str, Any]:
        system_prompt = """
You are an expert job description parser.
You work with ALL types of roles - technical,
non-technical, creative, managerial, and more.

Extract structured data and return ONLY valid JSON:
{
  "title": "exact job title",
  "seniority_level": "junior|mid|senior|lead",
  "min_experience": <years as float>,
  "education_requirement": "Bachelor|Master|PhD|Other",
  "required_skills": ["skill1", "skill2"],
  "preferred_skills": ["skill1", "skill2"],
  "must_have": ["concrete requirement 1"],
  "nice_to_have": ["concrete requirement 1"],
  "keywords": ["keyword1", "keyword2"]
}

Rules for ALL fields:
- required_skills: specific skills, tools, 
  competencies explicitly marked as required.
  For tech roles: React, Python, SQL etc.
  For non-tech roles: CRM, copywriting, 
  P&L management, campaign strategy etc.

- preferred_skills: skills mentioned as 
  preferred, bonus, or nice to have

- keywords: IMPORTANT - include ONLY meaningful
  domain-specific terms that would appear on a 
  strong candidate's resume. Include:
  * Tools and software (Salesforce, Figma, Excel)
  * Methodologies (Agile, Six Sigma, SEO, PPC)
  * Hard skills (financial modeling, A/B testing)
  * Domain terms (churn analysis, brand strategy)
  * Technical terms if technical role
  Maximum 15 keywords total.
  
  EXCLUDE from keywords:
  * Generic words: team, growing, talented, 
    looking, overview, requirements, join
  * Soft skills: communication, leadership,
    passionate, motivated, detail-oriented
  * Filler words: experience, professional,
    years, degree, related, field, strong

- min_experience: extract the minimum years 
  required. If range given (3-5 years), use 
  the minimum (3).

Return only JSON. No explanation. No markdown.
"""
        return self._call_llm_json(
          system_prompt=system_prompt,
          user_content=text,
          max_tokens=1000,
          label="parse_jd",
          validator=self._validate_jd_schema,
          default_payload={
            "title": "",
            "seniority_level": "junior",
            "min_experience": 0.0,
            "education_requirement": "Other",
            "required_skills": [],
            "preferred_skills": [],
            "must_have": [],
            "nice_to_have": [],
            "keywords": [],
          },
        )

  def evaluate_skills_match(
    self,
    resume_text: str,
    jd: dict,
    shared_context: Dict[str, Any] | None = None,
  ) -> dict:
        """
        LLM-powered skills evaluation that understands transferable and
        equivalent technologies across domains.

        Returns a dict with keys:
          skills_score, direct_matches, transferable_matches,
          missing_critical, reasoning
        """
        context = shared_context or build_llm_scoring_context(
          resume_text,
          jd,
          resume_excerpt_chars=4200,
        )
        system_prompt, user_content = build_skills_prompt(context)
        result = self._call_llm_json(
          system_prompt=system_prompt,
          user_content=user_content,
          max_tokens=1000,
          label="evaluate_skills_match",
          validator=self._validate_skills_schema,
          default_payload={
            "skills_score": 50.0,
            "direct_matches": [],
            "transferable_matches": [],
            "missing_critical": [],
            "reasoning": "Parse error",
          },
        )
        logger.info(f"LLM skills evaluation: score={result.get('skills_score', 50.0)}")
        return result

  def evaluate_project_quality(
    self,
    resume_text: str,
    jd: dict,
    shared_context: Dict[str, Any] | None = None,
  ) -> dict:
        """
        LLM-powered project quality evaluation that assesses impact,
        complexity, relevance, and candidate ownership.

        Returns a dict with keys:
          project_score, impact_score, complexity_score, relevance_score,
          ownership_score, best_project, reasoning
        """
        context = shared_context or build_llm_scoring_context(
          resume_text,
          jd,
          resume_excerpt_chars=4500,
        )
        system_prompt, user_content = build_projects_prompt(context)
        result = self._call_llm_json(
          system_prompt=system_prompt,
          user_content=user_content,
          max_tokens=600,
          label="evaluate_project_quality",
          validator=self._validate_projects_schema,
          default_payload={
            "project_score": 40.0,
            "impact_score": 40.0,
            "complexity_score": 40.0,
            "relevance_score": 40.0,
            "ownership_score": 40.0,
            "best_project": "",
            "reasoning": "Parse error",
          },
        )
        logger.info(f"LLM project evaluation: score={result.get('project_score', 40.0)}")
        return result

  def evaluate_experience_relevance(
    self,
    resume_text: str,
    jd: dict,
    shared_context: Dict[str, Any] | None = None,
  ) -> dict:
        """LLM-powered experience relevance evaluation that assesses domain
        match, seniority alignment, career trajectory, and recency.

        Returns a dict with keys:
          experience_score, domain_match_score, seniority_match_score,
          trajectory_score, relevant_years, employment_type, reasoning
        """
        context = shared_context or build_llm_scoring_context(
          resume_text,
          jd,
          resume_excerpt_chars=4200,
        )
        system_prompt, user_content = build_experience_prompt(context)
        result = self._call_llm_json(
          system_prompt=system_prompt,
          user_content=user_content,
          max_tokens=700,
          label="evaluate_experience_relevance",
          validator=self._validate_experience_schema,
          default_payload={
            "experience_score": 40.0,
            "domain_match_score": 40.0,
            "seniority_match_score": 40.0,
            "trajectory_score": 40.0,
            "relevant_years": 0.0,
            "employment_type": "unknown",
            "reasoning": "Parse error",
          },
        )
        logger.info(f"LLM experience evaluation: score={result.get('experience_score', 40.0)}")
        return result

  def evaluate_all_parallel(self, resume_text: str, jd: dict) -> dict:
        """
        Run evaluate_skills_match, evaluate_project_quality, and
        evaluate_experience_relevance in parallel using ThreadPoolExecutor.

        Returns combined dict:
        {
            "skills_score": float,
            "project_score": float,
            "experience_score": float,
            "skills_details": dict,
            "project_details": dict,
            "experience_details": dict
        }
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {
          "skills_score": 50.0,
          "project_score": 40.0,
          "experience_score": 40.0,
          "skills_details": {},
          "project_details": {},
          "experience_details": {},
        }

        parallel_start = time.time()
        shared_context = build_llm_scoring_context(
          resume_text,
          jd,
          resume_excerpt_chars=4500,
        )

        def run_skills():
            return (
              "skills",
              self.evaluate_skills_match(
                resume_text,
                jd,
                shared_context=shared_context,
              ),
            )

        def run_projects():
            return (
              "projects",
              self.evaluate_project_quality(
                resume_text,
                jd,
                shared_context=shared_context,
              ),
            )

        def run_experience():
            return (
              "experience",
              self.evaluate_experience_relevance(
                resume_text,
                jd,
                shared_context=shared_context,
              ),
            )

        tasks = [run_skills, run_projects, run_experience]

        with ThreadPoolExecutor(max_workers=3) as executor:
          futures = {
            executor.submit(task): task.__name__
            for task in tasks
          }
          for future in as_completed(futures):
            if time.time() - parallel_start > 90:
              logger.warning(
                "Parallel LLM evaluation exceeded 90s total timeout; cancelling remaining tasks."
              )
              for f in futures:
                f.cancel()
              break
            try:
              key, result = future.result(timeout=60)
              if key == "skills":
                results["skills_score"] = float(
                  result.get("skills_score", 50.0)
                )
                results["skills_details"] = result
              elif key == "projects":
                results["project_score"] = float(
                  result.get("project_score", 40.0)
                )
                results["project_details"] = result
              elif key == "experience":
                results["experience_score"] = float(
                  result.get("experience_score", 40.0)
                )
                results["experience_details"] = result
            except Exception as e:
              logger.warning(
                f"Parallel LLM eval failed for "
                f"{futures[future]}: {e}"
              )
              if isinstance(e, LLMAuthenticationException):
                raise

        logger.info(
          f"Parallel evaluation completed in "
          f"{time.time() - parallel_start:.1f}s"
        )

        log_event(
          "info",
          "parallel_llm_eval_complete",
          latency_seconds=f"{time.time() - parallel_start:.3f}",
          skills_score=results["skills_score"],
          project_score=results["project_score"],
          experience_score=results["experience_score"],
        )

        logger.info(
            f"Parallel LLM evaluation complete — "
            f"skills:{results['skills_score']} "
            f"projects:{results['project_score']} "
            f"experience:{results['experience_score']}"
        )
        return results

  def generate_strengths_gaps(
    self,
    resume: Dict[str, Any],
    jd: Dict[str, Any],
    scores: Dict[str, Any],
  ) -> Dict[str, Any]:
    system_prompt = """
You are an expert recruiter and talent assessor. You evaluate candidates for ALL types of roles - technical, non-technical, creative, managerial, and more.

Based on the resume, job description and scores provided, generate specific and honest strengths and gaps.

Return ONLY valid JSON:
{
  "strengths": [
    "Specific strength with evidence from resume",
    "Another specific strength",
    "Third specific strength"
  ],
  "gaps": [
    "Specific gap or missing requirement",
    "Another specific gap"
  ]
}

Rules:
- Strengths: reference SPECIFIC skills, achievements or experience from the resume that match what the JD needs. Be concrete.
  Good: "5 years React experience directly matches the required frontend skills"
  Bad: "Has good technical skills"

- Gaps: mention SPECIFIC requirements from the JD that are missing or weak in the resume
  Good: "No AWS experience mentioned despite it being a preferred skill"
  Bad: "Could improve technical skills"

- Always write from a recruiter perspective
- Keep each point to one clear sentence
- Be honest but constructive
- Works for any industry or role type

Return only JSON. No explanation. No markdown.
"""

    user_content = f"""
=== JOB DESCRIPTION ===
Title: {jd.get('title', '')}
Seniority: {jd.get('seniority_level', '')}
Required Skills: {jd.get('required_skills', [])}
Must Have: {jd.get('must_have', [])}
JD Context: {jd.get('raw_text', '')[:1500]}

=== CANDIDATE PROFILE ===
Name: {resume.get('candidate_name', '')}
Experience: {resume.get('years_experience', 0)} years
Relevant Experience: {resume.get('relevant_experience', 0)} years
Skills: {resume.get('parsed_skills', [])}
Education: {resume.get('education_level', '')}
Resume Context: {resume.get('raw_text', '')[:1500]}

=== AI SCORES ===
Skills Match: {scores.get('skills_score', 0)}%
ATS Score: {scores.get('ats_score', 0)}%
Project Quality: {scores.get('project_score', 0)}%
Experience Score: {scores.get('experience_score', 0)}%
Education Score: {scores.get('education_score', 0)}%
Final Score: {scores.get('final_score', 0)}%
Recommendation: {scores.get('recommendation', '')}
"""

    return self._call_llm_json(
      system_prompt=system_prompt,
      user_content=user_content,
      max_tokens=1000,
      label="generate_strengths_gaps",
      validator=self._validate_strengths_gaps_schema,
      default_payload={"strengths": [], "gaps": []},
    )
