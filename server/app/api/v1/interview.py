from fastapi import APIRouter, HTTPException, WebSocket
from app.api.deps import DBSession, CurrentUser, LLM
from app.models.requests import InterviewStartRequest, InterviewMessageRequest
from app.models.responses import InterviewSessionResponse, InterviewFeedback
from app.agents.workflows.interview import InterviewWorkflow, InterviewType
from datetime import datetime
import uuid

router = APIRouter()

# Store active sessions (in production, use Redis)
active_sessions: dict[str, dict] = {}


@router.post("/start", response_model=InterviewSessionResponse)
async def start_interview(
    request: InterviewStartRequest,
    db: DBSession,
    user: CurrentUser,
    llm: LLM,
):
    """Start a new interview session."""
    session_id = str(uuid.uuid4())
    
    # Map request interview type to workflow enum
    interview_type_map = {
        "behavioral": InterviewType.BEHAVIORAL,
        "technical": InterviewType.TECHNICAL,
        "system_design": InterviewType.SYSTEM_DESIGN,
        "dsa": InterviewType.TECHNICAL,  # DSA uses technical rubric
    }
    
    workflow = InterviewWorkflow(llm)
    result = await workflow.start(
        session_id=session_id,
        interview_type=interview_type_map.get(request.interview_type.value, InterviewType.BEHAVIORAL),
        target_role="Software Engineer",  # TODO: Get from user profile
        difficulty=request.difficulty,
        max_questions=5,
    )
    
    # Store session state
    active_sessions[session_id] = {
        "state": result,
        "workflow": workflow,
        "started_at": datetime.utcnow(),
    }
    
    return InterviewSessionResponse(
        session_id=session_id,
        interview_type=request.interview_type.value,
        status="active",
        messages=result.get("messages", []),
        started_at=datetime.utcnow(),
    )


@router.post("/message", response_model=InterviewSessionResponse)
async def send_message(
    request: InterviewMessageRequest,
    db: DBSession,
    user: CurrentUser,
    llm: LLM,
):
    """Send message in interview session."""
    session = active_sessions.get(request.session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    
    workflow: InterviewWorkflow = session["workflow"]
    current_state = session["state"]
    
    # Add answer and continue workflow
    result = await workflow.add_candidate_answer(current_state, request.message)
    
    # Update session
    active_sessions[request.session_id]["state"] = result
    
    return InterviewSessionResponse(
        session_id=request.session_id,
        interview_type=current_state.get("interview_type", "behavioral"),
        status=result.get("status", "active"),
        messages=result.get("messages", []),
        started_at=session["started_at"],
    )


@router.post("/{session_id}/end", response_model=InterviewFeedback)
async def end_interview(
    session_id: str,
    db: DBSession,
    user: CurrentUser,
    llm: LLM,
):
    """End interview and get feedback."""
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    
    state = session["state"]
    final_feedback = state.get("final_feedback", {})
    answer_scores = state.get("answer_scores", [])
    
    # Calculate scores
    avg_score = sum(s.get("score", 0) for s in answer_scores) / max(len(answer_scores), 1)
    
    # Cleanup session
    del active_sessions[session_id]
    
    return InterviewFeedback(
        overall_score=int(avg_score),
        communication_score=int(avg_score * 0.9),
        technical_score=int(avg_score * 1.1),
        strengths=final_feedback.get("key_points", ["Good problem-solving approach"])[:3],
        improvements=final_feedback.get("key_points", ["Consider edge cases"])[-2:],
        detailed_feedback=final_feedback.get("summary", "Interview completed successfully."),
    )


@router.websocket("/ws/{session_id}")
async def interview_websocket(
    websocket: WebSocket,
    session_id: str,
):
    """WebSocket for real-time interview."""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            
            session = active_sessions.get(session_id)
            if not session:
                await websocket.send_json({"error": "Session not found"})
                break
            
            workflow: InterviewWorkflow = session["workflow"]
            result = await workflow.add_candidate_answer(session["state"], message)
            active_sessions[session_id]["state"] = result
            
            await websocket.send_json({
                "type": "message",
                "messages": result.get("messages", [])[-2:],
                "status": result.get("status"),
            })
            
            if result.get("status") == "completed":
                await websocket.send_json({
                    "type": "complete",
                    "feedback": result.get("final_feedback"),
                })
                break
                
    except Exception as e:
        await websocket.close(code=1000, reason=str(e))
