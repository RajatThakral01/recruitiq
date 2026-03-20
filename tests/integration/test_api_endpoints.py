import io
import os
import sys
import types
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from src.core.domain.entities.score_result import ScoreResultEntity
from src.infrastructure.exceptions import LLMTimeoutException


os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost:5432/testdb")
os.environ.setdefault("GROK_API_KEY", "test-key")


class _FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, normalize_embeddings=True):
        return [[0.1, 0.2, 0.3] for _ in texts]


sys.modules.setdefault(
    "sentence_transformers",
    types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer),
)


@pytest.fixture
def client(monkeypatch):
    from src import main
    from src.adapters.inbound.api.routes import resume_routes as routes

    monkeypatch.setattr(main.database, "init_db", lambda: None)
    monkeypatch.setattr(main.database, "ping", lambda: True)

    state = {
        "jd": None,
        "resume": None,
        "scores": [],
        "jobs": set(),
        "job_status": {},
    }

    class FakePDF:
        def extract_text(self, path: str) -> str:
            return "python fastapi postgres docker"

    class FakeLLM:
        def __init__(self):
            self.fail_resume_parse = False

        def parse_resume(self, text):
            if self.fail_resume_parse:
                raise LLMTimeoutException("timeout")
            return {
                "candidate_name": "Test Candidate",
                "email": "test@example.com",
                "years_experience": 3.0,
                "relevant_experience": 2.0,
                "education_level": "Bachelor",
                "parsed_skills": ["python", "fastapi"],
                "projects": [],
            }

        def parse_jd(self, text):
            return {
                "title": "Backend Engineer",
                "seniority_level": "mid",
                "min_experience": 2.0,
                "education_requirement": "Bachelor",
                "required_skills": ["Python", "FastAPI"],
                "preferred_skills": ["Docker"],
                "must_have": ["Python"],
                "nice_to_have": ["Docker"],
                "keywords": ["python", "fastapi", "docker"],
            }

        def evaluate_all_parallel(self, resume_text, jd):
            return {
                "skills_score": 80.0,
                "project_score": 70.0,
                "experience_score": 75.0,
            }

        def generate_strengths_gaps(self, resume, jd, scores):
            return {"strengths": ["Strong Python"], "gaps": ["No Kubernetes"]}

    class FakeScoring:
        def get_quality_flag(self, text):
            return "high"

        def compute_all(self, resume, jd, resume_text, llm_scores=None):
            return ScoreResultEntity(
                id="score1",
                resume_id=resume.id,
                jd_id=jd.id,
                job_id="",
                skills_score=80.0,
                ats_score=60.0,
                project_score=70.0,
                experience_score=75.0,
                education_score=100.0,
                final_score=78.0,
                strengths=[],
                gaps=[],
                matched_keywords=["python"],
                missing_keywords=["fastapi"],
                keyword_match_rate=50.0,
                confidence_score=85.0,
                quality_flag="high",
                recommendation="Strong Fit",
                warning_flags={},
                processing_time_seconds=0.3,
                created_at=datetime.now(),
            )

    class FakeStorage:
        def save_jd(self, jd):
            state["jd"] = jd
            return jd.id

        def get_jd(self, jd_id):
            if state["jd"] is None:
                raise RuntimeError("jd not found")
            return state["jd"]

        def save_resume(self, resume):
            state["resume"] = resume
            return resume.id

        def get_resume(self, resume_id):
            return state["resume"]

        def create_screening_job(self, jd_id):
            job_id = "job-1"
            state["jobs"].add(job_id)
            state["job_status"][job_id] = "pending"
            return job_id

        def update_job_status(self, job_id, status):
            state["job_status"][job_id] = status
            return None

        def get_job_status(self, job_id):
            return state["job_status"].get(job_id)

        def save_score(self, score):
            state["scores"].append(score)
            return score.id

        def get_scores_by_job(self, job_id):
            return state["scores"]

        def job_exists(self, job_id):
            return job_id in state["jobs"]

    fake_llm = FakeLLM()
    monkeypatch.setattr(routes, "pdfplumber_adapter", FakePDF())
    monkeypatch.setattr(routes, "llm_adapter", fake_llm)
    monkeypatch.setattr(routes, "scoring_engine", FakeScoring())
    fake_storage = FakeStorage()
    monkeypatch.setattr(routes, "postgres_adapter", fake_storage)
    routes.parse_jd_use_case.llm_port = fake_llm
    routes.parse_jd_use_case.storage_port = fake_storage
    routes.rank_use_case.storage_port = fake_storage

    yield TestClient(main.app), fake_llm


def test_upload_jd_and_screen_results(client):
    test_client, _ = client

    jd_resp = test_client.post(
        "/api/upload/jd",
        files={"file": ("jd.txt", b"Backend role", "text/plain")},
    )
    assert jd_resp.status_code == 200
    jd_id = jd_resp.json()["jd_id"]

    resume_resp = test_client.post(
        "/api/upload/resumes",
        files=[("files", ("resume.pdf", io.BytesIO(b"fake-pdf"), "application/pdf"))],
    )
    assert resume_resp.status_code == 200
    resume_id = resume_resp.json()["resume_ids"][0]

    screen_resp = test_client.post(
        "/api/screen",
        json={"jd_id": jd_id, "resume_ids": [resume_id]},
    )
    assert screen_resp.status_code == 200
    payload = screen_resp.json()
    assert payload["status"] == "complete"
    assert payload["total_screened"] == 1

    results_resp = test_client.get(f"/api/results/{payload['job_id']}")
    assert results_resp.status_code == 200
    assert results_resp.json()["total_screened"] == 1


def test_upload_resumes_timeout_maps_to_504(client):
    test_client, fake_llm = client
    fake_llm.fail_resume_parse = True

    response = test_client.post(
        "/api/upload/resumes",
        files=[("files", ("resume.pdf", io.BytesIO(b"fake-pdf"), "application/pdf"))],
    )

    assert response.status_code == 504
    assert response.json()["code"] == "LLM_TIMEOUT"


def test_results_for_missing_job_returns_404(client):
    test_client, _ = client

    response = test_client.get("/api/results/nonexistent-job")

    assert response.status_code == 404
    payload = response.json()
    assert payload["code"] == "JOB_NOT_FOUND"


def test_upload_resumes_honors_configured_max(monkeypatch, client):
    test_client, _ = client

    from src.infrastructure.config import settings
    monkeypatch.setattr(settings, "MAX_RESUMES", 1)

    response = test_client.post(
        "/api/upload/resumes",
        files=[
            ("files", ("resume1.pdf", io.BytesIO(b"fake-pdf-1"), "application/pdf")),
            ("files", ("resume2.pdf", io.BytesIO(b"fake-pdf-2"), "application/pdf")),
        ],
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == "TOO_MANY_FILES"


def test_upload_jd_rejects_oversized_file(monkeypatch, client):
    test_client, _ = client

    from src.infrastructure.config import settings
    monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 0)

    response = test_client.post(
        "/api/upload/jd",
        files={"file": ("jd.txt", b"x", "text/plain")},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == "FILE_TOO_LARGE"


def test_upload_resume_rejects_oversized_file(monkeypatch, client):
    test_client, _ = client

    from src.infrastructure.config import settings
    monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 0)

    response = test_client.post(
        "/api/upload/resumes",
        files=[("files", ("resume.pdf", io.BytesIO(b"x"), "application/pdf"))],
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["code"] == "FILE_TOO_LARGE"
