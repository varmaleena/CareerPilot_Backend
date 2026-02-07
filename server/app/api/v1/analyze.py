from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.api.deps import DBSession, CurrentUser, LLM
from app.models.requests import AnalyzeRequest
from app.models.responses import AnalysisResponse
from app.agents.workflows.resume_analysis import ResumeAnalysisWorkflow
from datetime import datetime
import uuid

router = APIRouter()


@router.post("", response_model=AnalysisResponse)
async def analyze_resume(
    request: AnalyzeRequest,
    db: DBSession,
    user: CurrentUser,
    llm: LLM,
    background_tasks: BackgroundTasks,
):
    """Analyze resume and generate insights."""
    
    # Run analysis workflow
    workflow = ResumeAnalysisWorkflow(llm)
    result = await workflow.run(
        resume_text=request.resume_text,
        target_role=request.target_role,
        target_company=request.target_company,
    )
    
    # Extract analysis data from workflow result
    analysis_data = result.get("final_response", {}).get("analysis", {})
    content = analysis_data.get("content", "")
    key_points = analysis_data.get("key_points", [])
    
    # Generate analysis ID
    analysis_id = str(uuid.uuid4())
    
    return AnalysisResponse(
        analysis_id=analysis_id,
        readiness_score=75,  # TODO: Extract from actual analysis
        skill_gaps=[],
        strengths=key_points[:3] if key_points else ["Strong technical background"],
        weaknesses=["Areas for improvement identified"],
        recommendations=key_points[3:] if len(key_points) > 3 else ["Continue developing skills"],
        ats_score=80,
        created_at=datetime.utcnow(),
    )
