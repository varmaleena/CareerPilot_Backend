from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.api.deps import CurrentUser
from app.services.jobs.fetcher import job_fetcher, JobListing
from app.services.market.market_service import market_service

router = APIRouter()


class JobSearchRequest(BaseModel):
    query: str
    location: str = ""
    limit: int = 20


@router.post("/search")
async def search_jobs(
    request: JobSearchRequest,
    user: CurrentUser,
) -> list[JobListing]:
    """Search for jobs matching criteria."""
    return await job_fetcher.search(
        query=request.query,
        location=request.location,
        limit=request.limit,
    )


@router.post("/match")
async def match_jobs_to_resume(
    resume_data: dict,
    query: str,
    user: CurrentUser,
):
    """Find jobs that match the user's resume."""
    jobs = await job_fetcher.search(query=query, limit=20)
    matches = await job_fetcher.match_jobs_to_resume(resume_data, jobs)
    return matches


@router.get("/market/{role}")
async def get_market_insights(
    role: str,
    user: CurrentUser,
):
    """Get market insights for a specific role."""
    return await market_service.get_role_insights(role)


@router.post("/market/compare")
async def compare_skills(
    skills: list[str],
    target_role: str,
    user: CurrentUser,
):
    """Compare user skills to market requirements."""
    return await market_service.compare_skills_to_market(skills, target_role)


@router.get("/market/salary/{role}")
async def get_salary_benchmark(
    role: str,
    experience_years: int = Query(default=3, ge=0, le=30),
    user: CurrentUser = None,
):
    """Get salary benchmarks for a role."""
    return await market_service.salary_benchmark(role, experience_years)
