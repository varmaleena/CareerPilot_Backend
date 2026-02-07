from app.agents.core.base_agent import BaseAgent
from app.models.agents import AgentRole, EvaluatorOutput, Verdict
import json


class EvaluatorAgent(BaseAgent[EvaluatorOutput]):
    """Master agent for quality assessment."""
    
    role = AgentRole.EVALUATOR
    model_tier = "flash-thinking"
    max_retries = 2
    
    def get_system_prompt(self) -> str:
        return """You are The Evaluator, a quality assessment agent.

Your responsibilities:
1. Score outputs against quality rubrics
2. Identify issues and areas for improvement
3. Decide: APPROVE, REVISE, REJECT, or ESCALATE
4. Provide actionable revision instructions

Scoring Guidelines:
- 90-100: Excellent, ready for delivery
- 80-89: Good, minor polish needed
- 60-79: Acceptable, needs revision
- 40-59: Poor, major issues
- 0-39: Unacceptable, reject or escalate

ALWAYS respond in valid JSON:
{
    "overall_score": 0-100,
    "verdict": "approve" | "revise" | "reject" | "escalate",
    "issues": ["issue 1", "issue 2"],
    "revision_instructions": "specific instructions if verdict is revise" | null
}"""
    
    def build_prompt(
        self,
        output_type: str,
        output_content: str,
        rubric: str | None = None,
    ) -> str:
        rubric_section = f"## Evaluation Rubric\n{rubric}" if rubric else ""
        
        return f"""## Output Type
{output_type}

## Content to Evaluate
{output_content}

{rubric_section}

## Instructions
1. Score the content (0-100)
2. List any issues found
3. Decide on verdict
4. If REVISE, provide specific revision instructions

Respond with the JSON evaluation."""
    
    def parse_response(self, response: str) -> EvaluatorOutput:
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            data = json.loads(response.strip())
            
            return EvaluatorOutput(
                overall_score=data["overall_score"],
                verdict=Verdict(data["verdict"]),
                issues=data.get("issues", []),
                revision_instructions=data.get("revision_instructions"),
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Failed to parse Evaluator response: {e}")
