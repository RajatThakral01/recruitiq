import json
import time
from typing import Any, Dict

import openai
from src.core.ports.llm_port import LLMPort
from src.infrastructure.config import settings
from src.infrastructure.logger import logger
from src.infrastructure.exceptions import LLMTimeoutException, ScoringException


class ArceeAdapter(LLMPort):
  """Concrete implementation of LLMPort using Arcee AI (OpenAI-compatible) API."""

  def __init__(self) -> None:
    self.client = openai.OpenAI(
      api_key=settings.ARCEE_API_KEY,
      base_url=settings.ARCEE_BASE_URL,
    )
    self.model = settings.ARCEE_MODEL
    logger.info(f"ArceeAdapter initialized with model: {self.model}")

  def _safe_score(self, val: Any, default: float = 50.0) -> float:
    """Safely converts an LLM-provided score to a 0-100 float.

    If the value is missing or not a valid number, returns the provided default.
    """
    try:
      return max(0.0, min(100.0, float(val or default)))
    except (ValueError, TypeError):
      return default

  def _strip_markdown(self, text: str) -> str:
    """Strip markdown code fences from LLM response."""
    text = text.strip()
    if text.startswith("```"):
      lines = text.split("\n")
      lines = [l for l in lines if not l.strip().startswith("```")]
      text = "\n".join(lines).strip()
    return text

  def _call_arcee(self, system_prompt: str, user_content: str, max_tokens: int) -> str:
    """Helper to call Arcee with exponential backoff retry.
    Retries up to 3 times. Raises LLMTimeoutException on final failure.
    """
    for attempt in range(1, 4):
      try:
        logger.info(f"Arcee call attempt {attempt} – model={self.model}")
        response = self.client.chat.completions.create(
          model=self.model,
          messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
          ],
          max_tokens=max_tokens,
          temperature=0.0
        )
        content = response.choices[0].message.content
        if not content:
          raise ScoringException("Received empty response from Arcee AI.")
        return content.strip()
      except Exception as e:
        logger.error(f"Arcee call failed on attempt {attempt}: {e}")
        if attempt == 3:
          raise LLMTimeoutException("Arcee AI API call failed after 3 retries.", detail=str(e))
        time.sleep(2 ** attempt)
    raise LLMTimeoutException("Arcee AI call exhausted retries.")

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
    Convert total months to years rounded to 1 decimal.>,
  "relevant_experience": <float - ONLY count experience where the candidate's
    PRIMARY DAILY WORK directly matches the job type.

    Be STRICT and CONSERVATIVE:
    - Backend engineer role: only count months where candidate wrote backend
      code daily. Data analysis, frontend work, research = 0.
    - Data analyst role: only count months doing actual data analysis.
      Backend, ops, marketing = 0.
    - Marketing role: only count marketing work. Engineering, finance = 0.

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
  until today. Convert total months to years, round to 1 decimal.

- relevant_experience: STRICT domain match only.
  Full-time same domain = 100% credit.
  Internship same domain = 50% credit (0.5x months).
  Research same domain = 30% credit (0.3x months).
  Different domain entirely = 0.0 credit.
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
        try:
          raw = self._call_arcee(system_prompt, text, max_tokens=1000)
          parsed = json.loads(self._strip_markdown(raw))
          try:
            years_experience = max(
              0.0,
              float(parsed.get("years_experience", 0.0) or 0.0),
            )
          except (ValueError, TypeError):
            years_experience = 0.0
          parsed["years_experience"] = years_experience
          try:
            relevant_experience = max(
              0.0,
              float(parsed.get("relevant_experience", 0.0) or 0.0),
            )
          except (ValueError, TypeError):
            relevant_experience = 0.0
          parsed["relevant_experience"] = relevant_experience
          return parsed
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Arcee resume JSON: {e}")
            return {
                "candidate_name": "",
                "email": None,
                "years_experience": 0.0,
                "relevant_experience": 0.0,
                "education_level": "Other",
                "parsed_skills": [],
                "projects": [],
            }
        except Exception as e:
            logger.error(f"Error during resume parsing with Arcee: {e}")
            raise ScoringException("Resume parsing failed.", detail=str(e))

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
        try:
          raw = self._call_arcee(system_prompt, text, max_tokens=1000)
          return json.loads(self._strip_markdown(raw))
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Arcee JD JSON: {e}")
            return {
                "title": "",
                "seniority_level": "junior",
                "min_experience": 0.0,
                "education_requirement": "Other",
                "required_skills": [],
                "preferred_skills": [],
                "must_have": [],
                "nice_to_have": [],
                "keywords": [],
            }
        except Exception as e:
            logger.error(f"Error during JD parsing with Arcee: {e}")
            raise ScoringException("JD parsing failed.", detail=str(e))

  def evaluate_skills_match(self, resume_text: str, jd: dict) -> dict:
        """
        LLM-powered skills evaluation that understands transferable and
        equivalent technologies across domains.

        Returns a dict with keys:
          skills_score, direct_matches, transferable_matches,
          missing_critical, reasoning
        """
        system_prompt = """
You are a senior technical recruiter with a bias for action. Your goal is to find reasons to HIRE, not to disqualify. You believe in transferable skills and potential.

**CORE PHILOSOPHY: DOMAIN > TOOLS**
- A candidate's domain (e.g., backend, frontend, data science) is more important than the specific tools they've used.
- A strong backend engineer can learn any backend language. A strong frontend engineer can learn any frontend framework.
- **If the candidate is in the right domain, their base score is 70.** Do not start lower.

**EQUIVALENT SKILL GROUPS (Treat skills in the same group as 100% transferable):**
- **Backend:** Python, Java, Go, Node.js, C#, Rust, Ruby, PHP
- **Frontend:** React, Vue, Angular, Svelte, Next.js
- **Databases:** PostgreSQL, MySQL, MongoDB, DynamoDB, Cassandra
- **Cloud:** AWS, Google Cloud (GCP), Azure
- **DevOps:** Docker, Kubernetes, Terraform, Ansible
- **Data:** Pandas, NumPy, R, Spark, SQL

**SCORING METHODOLOGY (Follow these steps):**

1.  **Determine Domain Match:**
    - Is the candidate's experience in the same domain as the job? (e.g., Backend dev for Backend role).
    - **If YES, start with a base score of 70.**
    - If NO (e.g., QA for Backend role), start with a base score of 40.

2.  **Adjust Score Based on Skills:**
    - **Direct Matches:** For each required skill the candidate has *exactly*, **add +5 points**.
    - **Transferable Matches:** For each required skill where the candidate has an *equivalent* from the groups above, **add +3 points**.
    - **Critical Gaps:** If a "must-have" skill is truly missing and not transferable, **subtract -10 points**. (Use this sparingly).

3.  **Final Score:**
    - Calculate `base_score + points_from_direct_matches + points_from_transferable_matches - points_from_gaps`.
    - Clamp the final score between 0 and 100.
    - **A good candidate in the right domain should almost always score 70+.**

**EXAMPLE:**
- **JD:** Senior Node.js Engineer (Requires: Node.js, Docker, AWS).
- **Candidate:** 5 years of experience as a Python Backend Engineer, knows Docker and GCP.
- **Calculation:**
    - Domain match (Backend → Backend): **Base score = 70**.
    - Direct Matches (Docker): **+5 points**.
    - Transferable Matches (Python→Node.js, GCP→AWS): **+3 +3 = +6 points**.
    - Critical Gaps: None.
    - **Final Score: 70 + 5 + 6 = 81**.

Return ONLY a valid JSON object. No other text or explanations.
{
  "skills_score": <float>,
  "direct_matches": ["skill1", "skill2"],
  "transferable_matches": ["skill1→equivalent_skill"],
  "missing_critical": ["skill1"],
  "reasoning": "A brief, 2-sentence explanation for the score."
}
"""
        user_content = f"""
=== JOB DESCRIPTION ===
Title: {jd.get('title', '')}
Required Skills:
- {', '.join(jd.get('required_skills', []))}
Preferred Skills:
- {', '.join(jd.get('preferred_skills', []))}
Must Have Skills:
- {', '.join(jd.get('must_have', []))}

=== CANDIDATE RESUME ===
{resume_text[:4000]}
"""
        try:
          raw = self._call_arcee(system_prompt, user_content, max_tokens=1000)
          result = json.loads(self._strip_markdown(raw))
          result["skills_score"] = self._safe_score(
            result.get("skills_score", 50.0), default=50.0
          )
          score = result.get("skills_score", 50.0)
          logger.info(f"LLM skills evaluation: score={score}")
          return result
        except json.JSONDecodeError as e:
          logger.error(f"Failed to parse LLM skills evaluation JSON: {e}")
          return {
            "skills_score": 50.0,
            "direct_matches": [],
            "transferable_matches": [],
            "missing_critical": [],
            "reasoning": "Parse error",
          }
        except Exception as e:
          logger.error(f"LLM skills evaluation failed: {e}")
          raise ScoringException("LLM skills evaluation failed.", detail=str(e))

  def evaluate_project_quality(self, resume_text: str, jd: dict) -> dict:
        """
        LLM-powered project quality evaluation that assesses impact,
        complexity, relevance, and candidate ownership.

        Returns a dict with keys:
          project_score, impact_score, complexity_score, relevance_score,
          ownership_score, best_project, reasoning
        """
        system_prompt = """
You are a fair and experienced technical recruiter evaluating project work and achievements.

EVALUATION CRITERIA:

1. IMPACT (What did this accomplish?) - 30% of score
   ✓ Quantified: "Reduced latency by 40%", "Served 100K users", "$2M revenue"
   ✓ Qualitative: "Enabled new feature", "Shipped to production", "Migrated legacy system"
   ✓ Learning: "Built from scratch", "Novel approach to problem", "First time doing X"
   ✗ Don't penalize for missing metrics - technical work has value even without numbers
   
   Scoring:
   80-100: Measurable business impact (revenue, performance, scale) or shipped to production users
   60-79: Clear technical achievement (shipped features, completed projects, improved systems)
   40-59: Contributed to project, some visible results
   0-39: Minimal or unclear impact

2. COMPLEXITY (How technically deep?) - 25% of score
   ✓ Scale: "Handled 1M requests/day", "Multi-region deployment"
   ✓ Architecture: "Designed microservices", "Built data pipeline", "Led system redesign"
   ✓ Challenge: "First time with this tech", "Solved hard problem", "Novel algorithm"
   ✓ Learning: "Learned new domain", "Pioneered adoption of technology"
   
   Scoring:
   80-100: High-scale, complex architecture, novel solutions
   60-79: Medium complexity, non-trivial technical work
   40-59: Standard technical work
   0-39: Simple or routine tasks

3. RELEVANCE (Does it match the JD?) - 30% of score
   ✓ Same domain (backend→backend, frontend→frontend)
   ✓ Similar tech (Python backend work for Node.js role)
   ✓ Transferable skills (API design for microservices role)
   ✗ Don't require exact tech match - adjacent domains count
   
   Scoring:
   80-100: Directly matches JD requirements and role type
   60-79: Same domain, similar work (some tech differs but transferable)
   40-59: Related but different domain, some transferable skills
   0-39: Unrelated to JD requirements

4. OWNERSHIP (Who drove this?) - 15% of score
   ✓ Led, Built, Designed, Architected, Owned
   ✓ Drove, Implemented, Founded, Created
   ✓ "Responsible for", "Managed", "Directed"
   ~ Contributed to, Worked on (shared credit)
   ✗ Only assisted, helped, supported (low ownership)
   
   Scoring:
   80-100: Clear solo ownership or led a team
   60-79: Primary contributor on project
   40-59: Shared contribution, team member
   0-39: Minor contribution, mostly assisted

SCORING FORMULA:
project_score = (impact × 0.30) + (complexity × 0.25) + (relevance × 0.30) + (ownership × 0.15)

CRITICAL GUIDANCE:
✓ Give credit for what's explicitly stated
✓ Infer reasonable accomplishments (shipping = production = impact)
✓ If resume shows technical depth, score accordingly
✓ Don't penalize for missing quantitative metrics
✓ Score each dimension independently, then combine
✗ Don't require perfection - most projects score 50-80 range
✗ Only give <30 for clearly irrelevant work

Return ONLY valid JSON:
{
  "project_score": <float 0-100>,
  "impact_score": <float 0-100>,
  "complexity_score": <float 0-100>,
  "relevance_score": <float 0-100>,
  "ownership_score": <float 0-100>,
  "best_project": "name or description of strongest project",
  "reasoning": "2 sentence explanation of the score"
}

Return only JSON. No markdown. No explanations outside JSON.
"""
        user_content = f"""
=== JOB DESCRIPTION ===
Title: {jd.get('title', '')}
Domain/Level: {jd.get('seniority_level', '')} role
Key Requirements: {', '.join(jd.get('required_skills', []))}
Role Type: {jd.get('raw_text', '')[:500]}

=== CANDIDATE RESUME - EVALUATE PROJECTS & EXPERIENCE ===
{resume_text[:4500]}

=== EVALUATION CHECKLIST ===
1. IMPACT: What did candidate accomplish? Look for:
   - Shipped features/products to real users
   - Performance improvements (%, scale, metrics)
   - Revenue/business impact
   - Systems built from scratch
   - Migrations or major technical improvements

2. COMPLEXITY: How technically sophisticated?
   - Large-scale systems (100K users, high QPS)
   - Architectural decisions (microservices, databases, caching)
   - Novel technical solutions
   - Full-stack vs specialized depth

3. RELEVANCE: How related to this JD?
   - Same domain/role type (backend→backend, data→data, etc)
   - Similar tech stack
   - Transferable skills
   - Problem-solving patterns applicable to JD

4. OWNERSHIP: Did they lead or contribute?
   - Solo founder / solo builder = high ownership
   - Led team / owned subsystem = high ownership  
   - Co-founder / primary contributor = high ownership
   - Team member / contributor = medium ownership
   - Assisted/supported = lower ownership

Score each dimension fairly. Good candidates have 1-2 strong projects, not 5+ mediocre ones.
"""
        try:
          raw = self._call_arcee(system_prompt, user_content, max_tokens=600)
          result = json.loads(self._strip_markdown(raw))
          result["project_score"] = self._safe_score(
            result.get("project_score", 40.0), default=40.0
          )
          result["impact_score"] = self._safe_score(
            result.get("impact_score", 40.0), default=40.0
          )
          result["complexity_score"] = self._safe_score(
            result.get("complexity_score", 40.0), default=40.0
          )
          result["relevance_score"] = self._safe_score(
            result.get("relevance_score", 40.0), default=40.0
          )
          result["ownership_score"] = self._safe_score(
            result.get("ownership_score", 40.0), default=40.0
          )
          score = result.get("project_score", 40.0)
          logger.info(f"LLM project evaluation: score={score}")
          return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM project evaluation JSON: {e}")
            return {
                "project_score": 40.0,
                "impact_score": 40.0,
                "complexity_score": 40.0,
                "relevance_score": 40.0,
                "ownership_score": 40.0,
                "best_project": "",
                "reasoning": "Parse error",
            }
        except Exception as e:
            logger.error(f"LLM project evaluation failed: {e}")
            raise ScoringException("LLM project evaluation failed.", detail=str(e))

  def evaluate_experience_relevance(self, resume_text: str, jd: dict) -> dict:
        """LLM-powered experience relevance evaluation that assesses domain
        match, seniority alignment, career trajectory, and recency.

        Returns a dict with keys:
          experience_score, domain_match_score, seniority_match_score,
          trajectory_score, relevant_years, employment_type, reasoning
        """
        system_prompt = """
You are an expert recruiter evaluating how relevant a candidate's work
experience is to a specific job description.

Evaluate:
1. DOMAIN MATCH - Is work in the same field?
   (backend eng for backend role = high,
    data analyst for backend role = low)
2. SENIORITY MATCH - Does level match JD?
   (senior role needs senior signals)
3. TRAJECTORY - Is career progression positive?
   (growing responsibilities, promotions)
4. RECENCY - Is relevant experience recent?
   (last 1-2 years weighted more)
5. COMPANY CONTEXT - Quality of experience?
   (well-known companies, funded startups,
    research institutions = higher quality)

Return ONLY valid JSON:
{
  "experience_score": <float 0-100>,
  "domain_match_score": <float 0-100>,
  "seniority_match_score": <float 0-100>,
  "trajectory_score": <float 0-100>,
  "relevant_years": <float>,
  "employment_type": "fulltime|intern|mixed",
  "reasoning": "2 sentence explanation"
}

Scoring guide:
90-100: Highly relevant experience, perfect domain and seniority match
70-89:  Strong relevant experience with minor gaps
50-69:  Moderate relevance, some domain or seniority mismatch
30-49:  Limited relevance, significant domain mismatch
0-29:   Mostly irrelevant experience

experience_score = weighted average:
domain_match(40%) + seniority_match(25%) + trajectory(20%) + recency(15%)
Note: recency is implied from how you weight recent vs old experience.

Be strict about domain matching.
A data analyst applying for backend engineer should get
domain_match_score of 10-20, not 50+.
Return only JSON. No markdown.
"""
        user_content = (
          f"""
    === JOB DESCRIPTION ===
    Title: {jd.get('title', '')}
    Seniority Level: {jd.get('seniority_level', '')}
    Min Experience Required: {jd.get('min_experience', 0)} years
    Required Skills: {jd.get('required_skills', [])}
    Must Have Requirements: {jd.get('must_have', [])}
    Role Responsibilities:
    {jd.get('raw_text', '')[:2000]}

    === CANDIDATE RESUME ===
    {resume_text[:4000]}

    Evaluate whether the candidate's WORK HISTORY
    and CAREER TRAJECTORY match what this role needs.
    Be strict about domain relevance:
    - Same domain = high domain_match_score (70-100)
    - Adjacent domain = medium score (40-69)
    - Different domain = low score (10-39)
    """
        )
        try:
          raw = self._call_arcee(system_prompt, user_content, max_tokens=700)
          result = json.loads(self._strip_markdown(raw))
          result["experience_score"] = self._safe_score(
            result.get("experience_score", 40.0), default=40.0
          )
          result["domain_match_score"] = self._safe_score(
            result.get("domain_match_score", 40.0), default=40.0
          )
          result["seniority_match_score"] = self._safe_score(
            result.get("seniority_match_score", 40.0), default=40.0
          )
          result["trajectory_score"] = self._safe_score(
            result.get("trajectory_score", 40.0), default=40.0
          )
          score = result.get("experience_score", 40.0)
          logger.info(f"LLM experience evaluation: score={score}")
          return result
        except json.JSONDecodeError as e:
          logger.error(f"Failed to parse LLM experience evaluation JSON: {e}")
          return {
            "experience_score": 40.0,
            "domain_match_score": 40.0,
            "seniority_match_score": 40.0,
            "trajectory_score": 40.0,
            "relevant_years": 0.0,
            "employment_type": "unknown",
            "reasoning": "Parse error",
          }
        except Exception as e:
          logger.error(f"LLM experience evaluation failed: {e}")
          raise ScoringException("LLM experience evaluation failed.", detail=str(e))

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

        def run_skills():
            return ("skills", self.evaluate_skills_match(resume_text, jd))

        def run_projects():
            return ("projects", self.evaluate_project_quality(resume_text, jd))

        def run_experience():
            return ("experience", self.evaluate_experience_relevance(resume_text, jd))

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

        logger.info(
          f"Parallel evaluation completed in "
          f"{time.time() - parallel_start:.1f}s"
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

    try:
      raw = self._call_arcee(system_prompt, user_content, max_tokens=1000)
      return json.loads(self._strip_markdown(raw))
    except json.JSONDecodeError as e:
      logger.error(f"Failed to parse strengths/gaps JSON: {e}")
      return {"strengths": [], "gaps": []}
    except Exception as e:
      logger.error(f"Error during strengths/gaps generation: {e}")
      raise ScoringException("Strengths/gaps generation failed.", detail=str(e))
