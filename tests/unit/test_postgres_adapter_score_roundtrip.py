from contextlib import contextmanager
from datetime import datetime

from src.adapters.outbound.storage.postgres_adapter import PostgresAdapter
from src.core.domain.entities.score_result import ScoreResultEntity


class _FakeCursor:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params):
        self.executed.append((sql, params))

    def fetchone(self):
        return ["score-id-1"]

    def fetchall(self):
        return self.rows


class _FakeConnection:
    def __init__(self, rows=None):
        self.cursor_obj = _FakeCursor(rows=rows)

    def cursor(self):
        return self.cursor_obj


@contextmanager
def _fake_get_connection(rows=None):
    yield _FakeConnection(rows=rows)


def test_save_score_writes_warning_flags_and_metadata(monkeypatch):
    adapter = PostgresAdapter()
    monkeypatch.setattr(
        "src.adapters.outbound.storage.postgres_adapter.get_connection",
        lambda: _fake_get_connection(),
    )

    score = ScoreResultEntity(
        id="score-id-1",
        resume_id="r1",
        jd_id="j1",
        job_id="job1",
        skills_score=80.0,
        ats_score=75.0,
        project_score=70.0,
        experience_score=65.0,
        education_score=90.0,
        final_score=77.5,
        strengths=["s1"],
        gaps=["g1"],
        matched_keywords=["python"],
        missing_keywords=["fastapi"],
        keyword_match_rate=50.0,
        confidence_score=88.0,
        quality_flag="high",
        recommendation="Strong Fit",
        warning_flags={"projects_llm_fallback": "Used algorithmic fallback."},
        llm_provider="grok",
        llm_model="grok-2-latest",
        prompt_version="scoring-prompts-v1",
        processing_time_seconds=1.2,
        created_at=datetime.now(),
    )

    adapter.save_score(score)


def test_get_scores_by_job_reads_warning_flags_and_metadata(monkeypatch):
    now = datetime.now()
    rows = [
        (
            "score-id-1",
            "job1",
            "r1",
            "j1",
            80.0,
            75.0,
            70.0,
            65.0,
            90.0,
            77.5,
            ["s1"],
            ["g1"],
            ["python"],
            ["fastapi"],
            50.0,
            88.0,
            "high",
            "Strong Fit",
            {"projects_llm_fallback": "Used algorithmic fallback."},
            "grok",
            "grok-2-latest",
            "scoring-prompts-v1",
            1.2,
            now,
        )
    ]

    adapter = PostgresAdapter()
    monkeypatch.setattr(
        "src.adapters.outbound.storage.postgres_adapter.get_connection",
        lambda: _fake_get_connection(rows=rows),
    )

    result = adapter.get_scores_by_job("job1")

    assert len(result) == 1
    assert result[0].warning_flags["projects_llm_fallback"]
    assert result[0].llm_provider == "grok"
    assert result[0].llm_model == "grok-2-latest"
    assert result[0].prompt_version == "scoring-prompts-v1"
