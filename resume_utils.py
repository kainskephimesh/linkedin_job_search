"""
Resume parsing + job-fit scoring, powered by the same Azure OpenAI client
used elsewhere in this project (mcp_server._az_client).
"""
import asyncio
import io
import json

from pypdf import PdfReader
import docx

from mcp_server import _az_client, AZURE_DEPLOYMENT


def extract_text_from_resume(file_bytes: bytes, filename: str) -> str:
    """Extract raw text from an uploaded resume file (.pdf, .docx, or .txt)."""
    name = filename.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if name.endswith(".docx"):
        document = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in document.paragraphs)

    return file_bytes.decode("utf-8", errors="ignore")


_RESUME_PARSE_SYSTEM = """\
You are a resume parser for a job-search tool. Given raw resume text, extract the \
candidate's profile and return ONLY a valid JSON object with exactly these fields:
{
  "roles": ["job titles this candidate should search for, max 8 — their current/target \
title plus close synonyms actually used in job postings"],
  "locations": ["likely target locations, max 4 — from resume's address/current city, \
or [\\"Remote\\"] if none is stated"],
  "max_exp_years": <integer — candidate's total years of relevant professional experience>,
  "skills": ["key technical/professional skills, max 15"],
  "summary": "<one sentence summarizing the candidate's profile>"
}
Rules: infer sensibly from context (e.g. graduation year, job history) if experience \
isn't stated explicitly. Use realistic, specific job titles — not generic ones like "Engineer".
"""


async def parse_resume_with_llm(resume_text: str) -> dict:
    """Extract target roles/locations/experience/skills from resume text via LLM."""

    def _call() -> str:
        resp = _az_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _RESUME_PARSE_SYSTEM},
                {"role": "user", "content": resume_text[:6000]},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=600,
            temperature=0,
        )
        return resp.choices[0].message.content

    raw = await asyncio.to_thread(_call)
    return json.loads(raw)


_FIT_SCORE_SYSTEM = """\
You compare a candidate's resume against a single job posting for a job-search tool. \
Return ONLY a valid JSON object with exactly these fields:
{
  "fit_score": <integer 0-100, how well this candidate matches this specific job>,
  "reason": "<one short sentence: key matches and/or key gaps>"
}
Score based on: role/title alignment, required skills overlap, and experience-level fit. \
Be discriminating — most jobs should NOT score above 85 unless the match is genuinely strong.
"""


async def score_job_fit(resume_text: str, job: dict, semaphore: asyncio.Semaphore) -> dict:
    """Score how well a single scraped job matches the resume. Returns the job dict
    with 'fit_score' and 'fit_reason' added; falls back to a neutral score on failure."""

    job_text = (
        f"Role: {job.get('role', 'N/A')}\n"
        f"Company: {job.get('company', 'N/A')}\n"
        f"Location: {job.get('location', 'N/A')}\n"
        f"Experience required: {job.get('experience', 'N/A')}\n"
        f"Description: {job.get('snippet', '')}"
    )

    def _call() -> str:
        resp = _az_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": _FIT_SCORE_SYSTEM},
                {"role": "user", "content": f"RESUME:\n{resume_text[:4000]}\n\nJOB:\n{job_text}"},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=150,
            temperature=0,
        )
        return resp.choices[0].message.content

    async with semaphore:
        try:
            raw = await asyncio.to_thread(_call)
            scored = json.loads(raw)
            job = dict(job)
            job["fit_score"] = int(scored.get("fit_score", 50))
            job["fit_reason"] = scored.get("reason", "N/A")
        except Exception:
            job = dict(job)
            job["fit_score"] = 50
            job["fit_reason"] = "Could not be scored."
        return job


async def score_jobs_against_resume(resume_text: str, jobs: list[dict], max_concurrency: int = 5) -> list[dict]:
    """Score every job in parallel (capped concurrency) and return them sorted by fit_score desc."""
    semaphore = asyncio.Semaphore(max_concurrency)
    scored = await asyncio.gather(*[score_job_fit(resume_text, job, semaphore) for job in jobs])
    return sorted(scored, key=lambda j: j.get("fit_score", 0), reverse=True)
