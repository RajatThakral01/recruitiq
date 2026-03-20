from datetime import datetime

from src.core.domain.entities.job_description import JDEntity
from src.core.domain.entities.resume import ResumeEntity
from src.services.scoring_engine import ScoringEngine


class DummyATS:
    def score(self, resume_text, jd):
        return {
            "ats_score": 60.0,
            "matched_keywords": ["python"],
            "missing_keywords": ["fastapi"],
            "keyword_match_rate": 50.0,
        }


class CountingSemantic:
    def __init__(self):
        self.skills_calls = 0
        self.projects_calls = 0

    def score_skills(self, resume_text, jd):
        self.skills_calls += 1
        return 40.0

    def score_projects(self, resume_text, jd):
        self.projects_calls += 1
        return 50.0


def _resume() -> ResumeEntity:
    return ResumeEntity(
        id="r1",
        candidate_name="A",
        email="a@example.com",
        raw_text="python fastapi project",
        parsed_skills=["python"],
        years_experience=3.0,
        education_level="Bachelor",
        projects=[],
        quality_flag="high",
        relevant_experience=2.0,
        created_at=datetime.now(),
    )


def _jd() -> JDEntity:
    return JDEntity(
        id="j1",
        title="Backend Engineer",
        required_skills=["Python", "FastAPI"],
        preferred_skills=[],
        min_experience=2.0,
        education_requirement="Bachelor",
        keywords=["python", "fastapi"],
        must_have=["python"],
        nice_to_have=[],
        seniority_level="mid",
        raw_text="Need backend python engineer",
        created_at=datetime.now(),
    )


def test_llm_first_uses_llm_and_skips_algo(monkeypatch):
    from src.infrastructure import config

    monkeypatch.setattr(config.settings, "SCORING_MODE", "llm_first")

    semantic = CountingSemantic()
    engine = ScoringEngine(DummyATS(), semantic)

    result = engine.compute_all(
        _resume(),
        _jd(),
        "resume text",
        llm_scores={"skills": 91.0, "projects": 84.0, "experience": 79.0},
    )

    assert semantic.skills_calls == 0
    assert semantic.projects_calls == 0
    assert result.skills_score == 91.0
    assert result.project_score == 84.0
    assert result.experience_score == 79.0
    assert result.warning_flags == {}


def test_hybrid_blends_and_marks_missing_dimension(monkeypatch):
    from src.infrastructure import config

    monkeypatch.setattr(config.settings, "SCORING_MODE", "hybrid")

    semantic = CountingSemantic()
    engine = ScoringEngine(DummyATS(), semantic)

    result = engine.compute_all(
        _resume(),
        _jd(),
        "resume text",
        llm_scores={"skills": 80.0, "projects": None, "experience": 20.0},
    )

    assert semantic.skills_calls == 1
    assert semantic.projects_calls == 1
    assert result.skills_score == 64.0  # 0.6*80 + 0.4*40
    assert result.project_score == 50.0  # fallback to algo due to missing llm
    assert result.experience_score > 0
    assert "projects_llm_fallback" in result.warning_flags


def test_hybrid_marks_global_unavailable(monkeypatch):
    from src.infrastructure import config

    monkeypatch.setattr(config.settings, "SCORING_MODE", "hybrid")

    semantic = CountingSemantic()
    engine = ScoringEngine(DummyATS(), semantic)

    result = engine.compute_all(_resume(), _jd(), "resume text", llm_scores=None)

    assert semantic.skills_calls == 1
    assert semantic.projects_calls == 1
    assert "llm_scores_unavailable" in result.warning_flags
