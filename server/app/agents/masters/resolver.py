from app.agents.core.base_agent import BaseAgent
from app.models.agents import AgentRole
from pydantic import BaseModel
import json


class ResolverOutput(BaseModel):
    resolution: str
    confidence: int
    fallback_response: str | None
    should_notify_user: bool
    user_message: str | None


class ResolverAgent(BaseAgent[ResolverOutput]):
    """Master agent for conflict resolution and edge cases."""
    
    role = AgentRole.RESOLVER
    model_tier = "pro"
    max_retries = 1  # Critical path, don't retry much
    
    def get_system_prompt(self) -> str:
        return """You are The Resolver, the final decision-maker.

Your responsibilities:
1. Resolve conflicts between agent outputs
2. Handle edge cases that other agents can't process
3. Create graceful fallback responses
4. Decide when to involve the user

Principles:
- When in doubt, be conservative
- User trust is paramount
- Never make up information
- Always provide a usable response

ALWAYS respond in valid JSON:
{
    "resolution": "the final resolved output",
    "confidence": 0-100,
    "fallback_response": "simplified response if resolution fails" | null,
    "should_notify_user": true | false,
    "user_message": "message to show user if should_notify_user is true" | null
}"""
    
    def build_prompt(
        self,
        issue_type: str,
        conflicting_outputs: list[dict],
        context: dict | None = None,
    ) -> str:
        outputs_str = json.dumps(conflicting_outputs, indent=2)
        context_str = json.dumps(context, indent=2) if context else "None"
        
        return f"""## Issue Type
{issue_type}

## Conflicting Outputs
{outputs_str}

## Context
{context_str}

## Instructions
1. Analyze the conflicting outputs
2. Determine the best resolution
3. If resolution is uncertain, provide a safe fallback
4. If the user needs to be involved, set should_notify_user to true

Respond with the JSON resolution."""
    
    def parse_response(self, response: str) -> ResolverOutput:
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            
            data = json.loads(response.strip())
            
            return ResolverOutput(
                resolution=data["resolution"],
                confidence=data["confidence"],
                fallback_response=data.get("fallback_response"),
                should_notify_user=data.get("should_notify_user", False),
                user_message=data.get("user_message"),
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Failed to parse Resolver response: {e}")
