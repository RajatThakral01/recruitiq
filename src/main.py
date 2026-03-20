import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.adapters.inbound.api.routes.resume_routes import router
from src.infrastructure import database
from src.infrastructure.config import settings
from src.infrastructure.logger import logger

# ── App creation ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="RecruitIQ API",
    description="AI-powered resume screening and ranking system.",
    version="1.0.0",
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


# ── Lifecycle events ──────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Initialize database schema on application startup with robust error handling."""
    logger.info("RecruitIQ API starting up…")
    try:
        database.init_db()
        logger.info("RecruitIQ API started.")
    except Exception as e:
        logger.error(
            f"STARTUP FAILED: {e}\n"
            f"Check DATABASE_URL and ARCEE_API_KEY in your .env file",
        )
        raise


@app.on_event("shutdown")
def on_shutdown():
    """Cleanup on application shutdown."""
    logger.info("RecruitIQ API shutting down.")
