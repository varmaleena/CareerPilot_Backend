from pydantic import BaseModel
from enum import Enum
from datetime import datetime
from typing import Any

class AgentRole(str, Enum):
    STRATEGIST = "strategist"
    EVALUATOR = "evaluator"
    RESOLVER = "resolver"
    EXTRACTOR = "extractor"
    GENERATOR = "generator"
    VALIDATOR = "validator"
    FORMATTER = "formatter"

class Complexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

class Verdict(str, Enum):
    APPROVE = "approve"
    REVISE = "revise"
    REJECT = "reject"
    ESCALATE = "escalate"

class ExecutionStep(BaseModel):
    """Single step in execution plan."""
    step: int
    agent: AgentRole
    task: str
    fallback_if: str | None = None
    timeout_seconds: int = 30

class StrategistOutput(BaseModel):
    """Output from Strategist agent."""
    complexity: Complexity
    confidence: int  # 0-100
    execution_plan: list[ExecutionStep]
    needs_clarification: bool = False
    clarification_question: str | None = None

class EvaluatorOutput(BaseModel):
    """Output from Evaluator agent."""
    overall_score: int  # 0-100
    verdict: Verdict
    issues: list[str] = []
    revision_instructions: str | None = None

class AgentMessage(BaseModel):
    """Message passed between agents."""
    from_agent: AgentRole
    to_agent: AgentRole
    payload: dict[str, Any]
    timestamp: datetime
