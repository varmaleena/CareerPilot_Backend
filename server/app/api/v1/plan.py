from fastapi import APIRouter
from app.api.deps import DBSession, CurrentUser, LLM
from app.models.requests import PlanRequest
from app.models.responses import LearningPlanResponse
from app.agents.workflows.learning_plan import LearningPlanWorkflow
import uuid

router = APIRouter()


@router.post("", response_model=LearningPlanResponse)
async def generate_learning_plan(
    request: PlanRequest,
    db: DBSession,
    user: CurrentUser,
    llm: LLM,
):
    """Generate personalized learning plan."""
    
    # Run learning plan workflow
    workflow = LearningPlanWorkflow(llm)
    result = await workflow.run(
        user_id=user["id"],
        target_role=request.target_role,
        current_skills=request.current_skills,
        skill_gaps=[{"skill": s, "priority": "high"} for s in request.current_skills],
        available_hours_per_week=request.hours_per_week,
        deadline_weeks=request.timeline_weeks,
        learning_style="mixed",
    )
    
    final_plan = result.get("final_plan", {})
    
    return LearningPlanResponse(
        plan_id=str(uuid.uuid4()),
        weeks=final_plan.get("milestones", []),
        resources=final_plan.get("resources", []),
        milestones=final_plan.get("next_actions", [
            f"Start learning journey for {request.target_role}",
            "Complete first milestone in 2 weeks",
        ]),
    )
