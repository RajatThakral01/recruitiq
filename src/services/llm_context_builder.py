from typing import Any


SCORING_RUBRIC = {
    "weights": {
        "skills": 0.30,
        "ats": 0.15,
        "projects": 0.20,
        "experience": 0.25,
        "education": 0.10,
    },
    "recommendation_thresholds": {
        "strong_fit": 65,
        "moderate_fit": 45,
        "not_fit": 0,
    },
    "score_range": "0-100",
}


def build_llm_scoring_context(
    resume_text: str,
    jd: dict[str, Any],
    *,
    resume_excerpt_chars: int = 4500,
    jd_excerpt_chars: int = 2500,
) -> dict[str, Any]:
    """Build a canonical context payload for all LLM scoring calls.

    The same structured context is reused across skills, project, and experience
    evaluation so each call gets consistent JD/resume/rubric grounding.
    """
    resume_excerpt = (resume_text or "")[:resume_excerpt_chars]
    lower_resume = resume_excerpt.lower()

    title_terms = [
        w.strip().lower()
        for w in str(jd.get("title", "")).split()
        if len(w.strip()) >= 3
    ]
    required_terms = [
        str(s).strip().lower()
        for s in (jd.get("required_skills", []) or [])
        if str(s).strip()
    ]
    must_have_terms = [
        str(s).strip().lower()
        for s in (jd.get("must_have", []) or [])
        if str(s).strip()
    ]

    matched_title_terms = [w for w in title_terms if w in lower_resume]
    matched_required_terms = [w for w in required_terms if w in lower_resume]
    matched_must_have_terms = [w for w in must_have_terms if w in lower_resume]

    internship_signals = [
        "intern", "internship", "co-op", "coop", "trainee", "apprentice",
    ]
    fulltime_signals = [
        "full-time", "full time", "software engineer", "developer", "engineer", "manager",
    ]
    company_tier_signals = [
        "google", "microsoft", "amazon", "meta", "apple", "netflix", "uber", "airbnb",
        "flipkart", "swiggy", "zomato", "razorpay", "cred", "phonepe", "groww",
        "infosys", "tcs", "wipro", "hcl", "accenture", "deloitte",
    ]

    lines = [line.strip() for line in resume_excerpt.splitlines() if line.strip()]
    experience_lines = [
        line for line in lines
        if any(token in line.lower() for token in (
            "experience", "intern", "engineer", "developer", "manager", "analyst", "lead"
        ))
    ][:25]

    return {
        "job": {
            "title": jd.get("title", ""),
            "seniority_level": jd.get("seniority_level", ""),
            "min_experience": jd.get("min_experience", 0),
            "education_requirement": jd.get("education_requirement", "Other"),
            "required_skills": jd.get("required_skills", []),
            "preferred_skills": jd.get("preferred_skills", []),
            "must_have": jd.get("must_have", []),
            "nice_to_have": jd.get("nice_to_have", []),
            "keywords": jd.get("keywords", []),
            "raw_excerpt": (jd.get("raw_text", "") or "")[:jd_excerpt_chars],
        },
        "candidate": {
            "resume_excerpt": resume_excerpt,
            "experience_signals": {
                "matched_title_terms": matched_title_terms,
                "matched_required_skills": matched_required_terms,
                "matched_must_have": matched_must_have_terms,
                "internship_signal_count": sum(1 for s in internship_signals if s in lower_resume),
                "fulltime_signal_count": sum(1 for s in fulltime_signals if s in lower_resume),
                "company_tier_signal_count": sum(1 for s in company_tier_signals if s in lower_resume),
                "experience_lines": experience_lines,
            },
        },
        "project_rules": {
            "project_score_formula": {
                "impact": 0.30,
                "complexity": 0.25,
                "relevance": 0.30,
                "ownership": 0.15,
            },
            "guidance": [
                "Give credit for transferable skills and adjacent technologies.",
                "Do not require exact keyword matches for every required skill.",
                "Do not penalize heavily for missing numeric metrics if technical depth is clear.",
            ],
        },
        "system": {
            "rubric": SCORING_RUBRIC,
        },
    }
