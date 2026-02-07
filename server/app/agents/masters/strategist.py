from app.agents.core.base_agent import BaseAgent
from app.models.agents import AgentRole, StrategistOutput, ExecutionStep, Complexity
from pydantic import ValidationError
import json

class StrategistAgent(BaseAgent[StrategistOutput]):
    """Master agent for planning and complex reasoning."""
    
    role = AgentRole.STRATEGIST
    model_tier = "pro"
    max_retries = 2
    timeout_seconds = 60
    
    def get_system_prompt(self) -> str:
        return """You are The Strategist, a master planning agent.

Your responsibilities:
1. Analyze request complexity
2. Create execution plans
3. Handle ambiguous requests
4. Design fallback strategies

You coordinate helper agents (Extractor, Generator, Validator, Formatter).
Only create plans that can be executed by available agents.

ALWAYS respond in valid JSON matching this schema:
{
    "complexity": "simple" | "moderate" | "complex",
    "confidence": 0-100,
    "execution_plan": [
        {
            "step": 1,
            "agent": "extractor" | "generator" | "validator" | "formatter",
            "task": "description of task",
            "fallback_if": "condition for fallback" | null
        }
    ],
    "needs_clarification": true | false,
    "clarification_question": "question if needs_clarification is true" | null
}"""
    
    def build_prompt(
        self,
        request_type: str,
        user_input: str,
        context: dict | None = None,
    ) -> str:
        context_str = json.dumps(context) if context else "None"
        
        return f"""## Request Type
{request_type}

## User Input
{user_input}

## Context
{context_str}

## Instructions
1. Assess the complexity of this request
2. Create an execution plan using available helper agents
3. If the request is ambiguous, set needs_clarification to true
4. For each step, consider what could go wrong and set fallback_if

Respond with the JSON plan."""
    
    def parse_response(self, response: str) -> StrategistOutput:
        # Extract JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            data = json.loads(response.strip())
            
            # Build execution steps
            steps = [
                ExecutionStep(
                    step=s["step"],
                    agent=s["agent"],
                    task=s["task"],
                    fallback_if=s.get("fallback_if"),
                )
                for s in data.get("execution_plan", [])
            ]
            
            return StrategistOutput(
                complexity=Complexity(data["complexity"]),
                confidence=data["confidence"],
                execution_plan=steps,
                needs_clarification=data.get("needs_clarification", False),
                clarification_question=data.get("clarification_question"),
            )
        except (json.JSONDecodeError, KeyError, ValidationError) as e:
            raise ValueError(f"Failed to parse Strategist response: {e}")
