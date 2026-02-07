from pydantic import BaseModel, Field, EmailStr
from enum import Enum


class InterviewType(str, Enum):
    BEHAVIORAL = "behavioral"
    TECHNICAL = "technical"
    SYSTEM_DESIGN = "system_design"
    DSA = "dsa"


class AnalyzeRequest(BaseModel):
    """Resume analysis request."""
    resume_text: str = Field(..., min_length=100, max_length=50000)
    target_role: str = Field(..., min_length=2, max_length=200)
    target_company: str | None = Field(None, max_length=200)

    class Config:
        json_schema_extra = {
            "example": {
                "resume_text": "Software Engineer with 5 years...",
                "target_role": "Senior Software Engineer",
                "target_company": "Google"
            }
        }


class InterviewStartRequest(BaseModel):
    """Start interview session."""
    interview_type: InterviewType
    difficulty: str = Field("medium", pattern="^(easy|medium|hard)$")
    duration_minutes: int = Field(30, ge=10, le=60)


class InterviewMessageRequest(BaseModel):
    """Send message in interview."""
    session_id: str
    message: str = Field(..., min_length=1, max_length=5000)


class PlanRequest(BaseModel):
    """Learning plan generation request."""
    target_role: str
    current_skills: list[str] = Field(default_factory=list)
    timeline_weeks: int = Field(12, ge=4, le=52)
    hours_per_week: int = Field(10, ge=5, le=40)


class ResumeOptimizeRequest(BaseModel):
    """Resume optimization request."""
    resume_text: str = Field(..., min_length=100)
    target_role: str
    optimization_focus: list[str] = Field(
        default_factory=lambda: ["ats_keywords", "impact_metrics"]
    )
