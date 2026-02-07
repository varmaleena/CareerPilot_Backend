from app.agents.core.base_agent import BaseAgent
from app.models.agents import AgentRole
from pydantic import BaseModel
import json


class ValidationResult(BaseModel):
    is_valid: bool
    document_type: str  # resume, cover_letter, unknown
    confidence: int
    issues: list[str]
    suggestions: list[str]


class ValidatorAgent(BaseAgent[ValidationResult]):
    """Helper agent for validation and classification."""
    
    role = AgentRole.VALIDATOR
    model_tier = "lite"
    max_retries = 2
    timeout_seconds = 5
    
    def get_system_prompt(self) -> str:
        return """You are a Validator agent. Your job is to validate inputs and classify documents.

Validation rules:
- Check for required content
- Identify document type
- Flag potential issues
- Suggest improvements

Respond with JSON:
{
    "is_valid": true|false,
    "document_type": "resume"|"cover_letter"|"job_description"|"unknown",
    "confidence": 0-100,
    "issues": ["issue1"],
    "suggestions": ["suggestion1"]
}"""
    
    def build_prompt(
        self,
        content: str,
        expected_type: str | None = None,
    ) -> str:
        expected = f"Expected type: {expected_type}" if expected_type else ""
        
        return f"""## Content to Validate
{content[:3000]}

{expected}

Analyze the content and respond with JSON validation result."""
    
    def parse_response(self, response: str) -> ValidationResult:
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            
            data = json.loads(response.strip())
            return ValidationResult(**data)
        except Exception as e:
            raise ValueError(f"Failed to parse validation: {e}")
