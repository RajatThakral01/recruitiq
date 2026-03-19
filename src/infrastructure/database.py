import contextlib
import psycopg2
from psycopg2 import pool
from src.infrastructure.config import settings
from src.infrastructure.logger import logger
from src.infrastructure.exceptions import DatabaseException

class DatabaseManager:
    """
    Manages PostgreSQL connection pooling using psycopg2 ThreadedConnectionPool.
    """
    _pool: pool.ThreadedConnectionPool = None

    @classmethod
    def get_pool(cls):
        """
        Initializes and returns the connection pool.
        """
        if cls._pool is None:
            try:
                cls._pool = pool.ThreadedConnectionPool(
                    1, 10, dsn=settings.DATABASE_URL
                )
                logger.info("Database connection pool initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize database pool: {str(e)}")
                raise DatabaseException("Could not connect to database.", detail=str(e))
        return cls._pool

    @classmethod
    @contextlib.contextmanager
    def get_connection(cls):
        """
        Context manager for getting a connection from the pool.
        Automatically handles commit, rollback, and returns connection to pool.
        """
        connection_pool = cls.get_pool()
        connection = connection_pool.getconn()
        try:
            yield connection
            connection.commit()
        except Exception as e:
            connection.rollback()
            logger.error(f"Database error occurred: {str(e)}")
            raise DatabaseException("Database transaction failed.", detail=str(e))
        finally:
            connection_pool.putconn(connection)

def init_db():
    """
    Initializes the database schema if tables do not exist.
    """
    schema_sql = """
    -- Enable UUID extension if not already present (built-in for PG 13+)
    CREATE EXTENSION IF NOT EXISTS "pgcrypto";

    CREATE TABLE IF NOT EXISTS job_descriptions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        title VARCHAR(255),
        required_skills JSONB,
        preferred_skills JSONB,
        keywords JSONB,
        must_have JSONB,
        nice_to_have JSONB,
        seniority_level VARCHAR(50),
        min_experience FLOAT,
        education_requirement VARCHAR(100),
        raw_text TEXT,
        created_at TIMESTAMP DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS resumes (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        candidate_name VARCHAR(255),
        email VARCHAR(255),
        raw_text TEXT,
        parsed_skills JSONB,
        years_experience FLOAT,
        education_level VARCHAR(100),
        projects JSONB,
        quality_flag VARCHAR(20),
        created_at TIMESTAMP DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS screening_jobs (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        jd_id UUID REFERENCES job_descriptions(id),
        status VARCHAR(20) DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT NOW(),
        completed_at TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS score_results (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        job_id UUID REFERENCES screening_jobs(id),
        resume_id UUID REFERENCES resumes(id),
        jd_id UUID REFERENCES job_descriptions(id),
        skills_score FLOAT,
        ats_score FLOAT,
        project_score FLOAT,
        experience_score FLOAT,
        education_score FLOAT,
        final_score FLOAT,
        strengths JSONB,
        gaps JSONB,
        matched_keywords JSONB,
        missing_keywords JSONB,
        keyword_match_rate FLOAT,
        confidence_score FLOAT,
        quality_flag VARCHAR(20),
        recommendation VARCHAR(20),
        processing_time_seconds FLOAT,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    try:
        with DatabaseManager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(schema_sql)
        logger.info("Database schema initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database schema: {str(e)}")
        raise DatabaseException("Could not initialize database.", detail=str(e))

def ping() -> bool:
    """
    Health check to verify database connectivity.
    """
    try:
        with DatabaseManager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True
    except Exception:
        return False

# Export convenient access to connection management
get_connection = DatabaseManager.get_connection
