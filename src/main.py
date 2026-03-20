import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.adapters.inbound.api.routes.resume_routes import router
from src.infrastructure import database
from src.infrastructure.config import settings
from src.infrastructure.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and teardown application resources."""
    logger.info("RecruitIQ API starting up…")
    try:
        database.init_db()
        logger.info("RecruitIQ API started.")
    except Exception as e:
        provider = (settings.LLM_PROVIDER or "mistral").strip().lower()
        if provider == "mistral":
            expected_key = "MISTRAL_API_KEY (fallback: GROQ_API_KEY or GROK_API_KEY)"
        elif provider == "groq":
            expected_key = "GROQ_API_KEY (fallback: MISTRAL_API_KEY or GROK_API_KEY)"
        else:
            expected_key = "GROK_API_KEY (fallback: MISTRAL_API_KEY or GROQ_API_KEY)"
        logger.error(
            f"STARTUP FAILED: {e}\n"
            f"Check DATABASE_URL and {expected_key} in your .env/.env.example file",
        )
        raise

    try:
        yield
    finally:
        logger.info("RecruitIQ API shutting down.")

# ── App creation ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="RecruitIQ API",
    description="AI-powered resume screening and ranking system.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
origins = [o.strip() for o in settings.CORS_ORIGINS.split(",")] if settings.CORS_ORIGINS else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static assets ────────────────────────────────────────────────────────────
# Try to figure out where the frontend directory actually is (handle the "frontend:" edge case)
frontend_dir = "frontend"
if not os.path.exists(os.path.join(frontend_dir, "code.html")):
    if os.path.exists("frontend/frontend/code.html"):
        frontend_dir = "frontend/frontend"
    elif os.path.exists("frontend:/code.html"):
        frontend_dir = "frontend:"

if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# ── API routes ────────────────────────────────────────────────────────────────
app.include_router(router)


# ── Root → serve SPA ─────────────────────────────────────────────────────────
@app.get("/")
async def serve_frontend():
    """Serves the frontend HTML file with fallback logic."""
    # Check standard paths
    paths_to_check = [
        Path("frontend/code.html"),
        Path("frontend/frontend/code.html"),
        Path("frontend:/code.html"),  # Handle the colon directory if present
    ]

    for file_path in paths_to_check:
        if file_path.exists():
            return FileResponse(str(file_path))

    return {
        "error": "Frontend file not found",
        "looked_at": [str(p.absolute()) for p in paths_to_check],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
