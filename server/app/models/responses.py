from pydantic import BaseModel
from datetime import datetime
from typing import Any


class SkillGap(BaseModel):
    skill: str
    current_level: int  # 0-100
    required_level: int
    priority: str  # high, medium, low


class AnalysisResponse(BaseModel):
    """Resume analysis response."""
    analysis_id: str
    readiness_score: int  # 0-100
    skill_gaps: list[SkillGap]
    strengths: list[str]
    weaknesses: list[str]
    recommendations: list[str]
    ats_score: int
    created_at: datetime


class InterviewSessionResponse(BaseModel):
    """Interview session info."""
    session_id: str
    interview_type: str
    status: str
    messages: list[dict[str, Any]]
    started_at: datetime


class InterviewFeedback(BaseModel):
    """Interview feedback response."""
    overall_score: int
    communication_score: int
    technical_score: int
    strengths: list[str]
    improvements: list[str]
    detailed_feedback: str


class LearningPlanResponse(BaseModel):
    """Learning plan response."""
    plan_id: str
    weeks: list[dict[str, Any]]
    resources: list[dict[str, Any]]
    milestones: list[str]


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str | None = None
    code: str | None = None
