from app.agents.core.base_agent import BaseAgent
from app.models.agents import AgentRole
from pydantic import BaseModel
import json


class FormattedOutput(BaseModel):
    formatted_content: str
    format_type: str
    character_count: int


class FormatterAgent(BaseAgent[FormattedOutput]):
    """Helper agent for output formatting."""
    
    role = AgentRole.FORMATTER
    model_tier = "lite"
    max_retries = 2
    timeout_seconds = 10
    
    def get_system_prompt(self) -> str:
        return """You are a Formatter agent. Your job is to format content for delivery.

Formatting guidelines:
- Use markdown for rich text
- Use LaTeX for resumes when requested
- Keep formatting clean and readable
- Preserve all important information

Respond with JSON:
{
    "formatted_content": "the formatted output",
    "format_type": "markdown"|"latex"|"json"|"plain",
    "character_count": 123
}"""
    
    def build_prompt(
        self,
        content: str,
        target_format: str,
        style_guide: str | None = None,
    ) -> str:
        style = f"Style Guide: {style_guide}" if style_guide else ""
        
        return f"""## Content
{content}

## Target Format
{target_format}

{style}

Format the content and respond with JSON."""
    
    def parse_response(self, response: str) -> FormattedOutput:
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            
            data = json.loads(response.strip())
            return FormattedOutput(**data)
        except Exception as e:
            raise ValueError(f"Failed to parse formatted output: {e}")
