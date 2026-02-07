from app.agents.core.base_agent import BaseAgent
from app.models.agents import AgentRole
from pydantic import BaseModel
import json


class GeneratedContent(BaseModel):
    content: str
    content_type: str
    word_count: int
    key_points: list[str]


class GeneratorAgent(BaseAgent[GeneratedContent]):
    """Helper agent for content generation."""
    
    role = AgentRole.GENERATOR
    model_tier = "flash"
    max_retries = 2
    timeout_seconds = 30
    
    def get_system_prompt(self) -> str:
        return """You are a Generator agent. Your job is to create high-quality content.

Guidelines:
- Be specific and actionable
- Use clear, professional language
- Structure content logically
- Include concrete examples where helpful

Respond with JSON:
{
    "content": "the generated content",
    "content_type": "analysis|plan|feedback|question",
    "word_count": 123,
    "key_points": ["point1", "point2"]
}"""
    
    def build_prompt(
        self,
        generation_type: str,
        context: dict,
        requirements: list[str] | None = None,
    ) -> str:
        context_str = json.dumps(context, indent=2)
        reqs = "\n".join(f"- {r}" for r in (requirements or []))
        
        return f"""## Generation Type
{generation_type}

## Context
{context_str}

## Requirements
{reqs if reqs else "None specified"}

Generate the requested content and respond with JSON."""
    
    def parse_response(self, response: str) -> GeneratedContent:
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            
            data = json.loads(response.strip())
            return GeneratedContent(**data)
        except Exception as e:
            raise ValueError(f"Failed to parse generated content: {e}")
