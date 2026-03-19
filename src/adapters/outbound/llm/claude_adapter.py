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
You are a resume parser. Extract structured data 
from the resume text and return ONLY valid JSON 
with these exact fields:
{
  "candidate_name": "string",
  "email": "string or null",
  "years_experience": float,
  "education_level": "Bachelor|Master|PhD|Other",
  "parsed_skills": ["skill1", "skill2"],
  "projects": [
    {
      "name": "string",
      "description": "string",
      "tech_stack": ["tech1"],
      "metrics": "string or null"
    }
  ]
}
Return only JSON. No explanation. No markdown.
"""
        try:
            raw = self._call_arcee(system_prompt, text, max_tokens=1000)
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Arcee resume JSON: {e}")
            return {
                "candidate_name": "",
                "email": None,
                "years_experience": 0.0,
                "education_level": "Other",
                "parsed_skills": [],
                "projects": [],
            }
        except Exception as e:
            logger.error(f"Error during resume parsing with Arcee: {e}")
            raise ScoringException("Resume parsing failed.", detail=str(e))

    def parse_jd(self, text: str) -> Dict[str, Any]:
        system_prompt = """
You are a job description parser. Extract structured 
data and return ONLY valid JSON with these exact fields:
{
  "title": "string",
  "seniority_level": "junior|mid|senior|lead",
  "min_experience": float,
  "education_requirement": "Bachelor|Master|PhD|Other",
  "required_skills": ["skill1", "skill2"],
  "preferred_skills": ["skill1", "skill2"],
  "must_have": ["requirement1"],
  "nice_to_have": ["requirement1"],
  "keywords": ["keyword1", "keyword2"]
}
Return only JSON. No explanation. No markdown.
"""
        try:
            raw = self._call_arcee(system_prompt, text, max_tokens=1000)
            return json.loads(raw)
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

    def generate_strengths_gaps(self, resume: Dict[str, Any], jd: Dict[str, Any], scores: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = """
You are an expert recruiter AI. Based on the resume, 
job description and scores provided, generate 
concise strengths and gaps.
Return ONLY valid JSON:
{
  "strengths": [
    "One sentence strength 1",
    "One sentence strength 2", 
    "One sentence strength 3"
  ],
  "gaps": [
    "One sentence gap 1",
    "One sentence gap 2"
  ]
}
Be specific. Reference actual skills and experience.
No generic statements. No markdown. JSON only.
"""
        user_content = json.dumps({"resume": resume, "jd": jd, "scores": scores})
        try:
            raw = self._call_arcee(system_prompt, user_content, max_tokens=1000)
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse strengths/gaps JSON: {e}")
            return {"strengths": [], "gaps": []}
        except Exception as e:
            logger.error(f"Error during strengths/gaps generation: {e}")
            raise ScoringException("Strengths/gaps generation failed.", detail=str(e))
