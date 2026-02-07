from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.api.deps import DBSession, CurrentUser, LLM
from app.agents.workflows.resume_optimization import ResumeOptimizationWorkflow
import uuid

router = APIRouter()


class OptimizeRequest(BaseModel):
    resume_text: str = Field(..., min_length=100)
    job_description: str = Field(..., min_length=50)
    target_role: str
    optimization_focus: list[str] = Field(
        default_factory=lambda: ["ats", "bullets", "keywords"]
    )


class OptimizeResponse(BaseModel):
    optimization_id: str
    optimized_resume: str
    ats_score_before: int
    ats_score_after: int
    improvement: int
    changes_made: list[dict]
    suggestions: list[str]


@router.post("", response_model=OptimizeResponse)
async def optimize_resume(
    request: OptimizeRequest,
    db: DBSession,
    user: CurrentUser,
    llm: LLM,
):
    """Optimize resume for ATS and target job."""
    
    workflow = ResumeOptimizationWorkflow(llm)
    result = await workflow.run(
        resume_text=request.resume_text,
        job_description=request.job_description,
        target_role=request.target_role,
        optimization_focus=request.optimization_focus,
    )
    
    final_report = result.get("final_report", {})
    
    return OptimizeResponse(
        optimization_id=str(uuid.uuid4()),
        optimized_resume=result.get("optimized_resume", ""),
        ats_score_before=final_report.get("ats_score_before", 0),
        ats_score_after=final_report.get("ats_score_after", 0),
        improvement=final_report.get("improvement", 0),
        changes_made=result.get("changes_made", []),
        suggestions=result.get("suggested_additions", []),
    )
