from fastapi import APIRouter
from app.api.v1 import analyze, plan, interview, optimize, export, jobs

api_router = APIRouter()

api_router.include_router(analyze.router, prefix="/analyze", tags=["Analysis"])
api_router.include_router(optimize.router, prefix="/optimize", tags=["Optimization"])
api_router.include_router(plan.router, prefix="/plan", tags=["Learning Plan"])
api_router.include_router(interview.router, prefix="/interview", tags=["Interview"])
api_router.include_router(export.router, prefix="/export", tags=["Export"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["Jobs & Market"])
