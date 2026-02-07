import httpx
from typing import Literal
from pydantic import BaseModel
from app.config import settings


class JobListing(BaseModel):
    title: str
    company: str
    location: str
    description: str
    salary_range: str | None = None
    url: str
    source: str
    posted_date: str | None = None


class JobFetcher:
    """Fetch job listings from various sources."""
    
    # API endpoints (placeholder URLs)
    SOURCES = {
        "linkedin": "https://api.linkedin.com/v2/jobs",
        "indeed": "https://api.indeed.com/v2/search",
        "glassdoor": "https://api.glassdoor.com/v1/jobs",
    }
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30)
    
    async def search(
        self,
        query: str,
        location: str = "",
        source: Literal["linkedin", "indeed", "glassdoor", "all"] = "all",
        limit: int = 20,
    ) -> list[JobListing]:
        """Search for jobs across sources."""
        jobs = []
        
        # For now, return mock data
        # In production, implement actual API calls
        mock_jobs = [
            JobListing(
                title=f"{query} Engineer",
                company="Tech Company",
                location=location or "Remote",
                description=f"Looking for a skilled {query} engineer...",
                salary_range="$100k - $150k",
                url="https://example.com/job/1",
                source="mock",
                posted_date="2024-01-15",
            ),
            JobListing(
                title=f"Senior {query} Developer",
                company="Startup Inc",
                location=location or "San Francisco, CA",
                description=f"Join our team as a senior {query} developer...",
                salary_range="$120k - $180k",
                url="https://example.com/job/2",
                source="mock",
                posted_date="2024-01-14",
            ),
        ]
        
        jobs.extend(mock_jobs[:limit])
        return jobs
    
    async def get_job_details(self, job_url: str) -> dict:
        """Fetch full job description from URL."""
        try:
            response = await self.client.get(job_url)
            # Parse and return job details
            return {
                "url": job_url,
                "full_description": "Full job description would be scraped here",
                "requirements": [],
                "benefits": [],
            }
        except Exception:
            return {"error": "Failed to fetch job details"}
    
    async def match_jobs_to_resume(
        self,
        resume_data: dict,
        jobs: list[JobListing],
    ) -> list[dict]:
        """Score job matches based on resume skills."""
        matches = []
        resume_skills = set(s.lower() for s in resume_data.get("skills", []))
        
        for job in jobs:
            # Simple keyword matching (LLM would be better)
            score = 0
            job_text = f"{job.title} {job.description}".lower()
            
            for skill in resume_skills:
                if skill in job_text:
                    score += 10
            
            matches.append({
                "job": job.model_dump(),
                "match_score": min(score, 100),
                "matched_skills": [s for s in resume_skills if s in job_text],
            })
        
        return sorted(matches, key=lambda x: x["match_score"], reverse=True)


job_fetcher = JobFetcher()
