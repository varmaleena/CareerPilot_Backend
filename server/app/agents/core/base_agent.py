from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic
from pydantic import BaseModel
# from app.services.llm.gateway import LLMGateway # Cyclic dependency - will inject at runtime or modify import
# For now, mocking the type hint to avoid import error until service is ready
from app.models.agents import AgentRole
from loguru import logger

T = TypeVar("T", bound=BaseModel)

class BaseAgent(ABC, Generic[T]):
    """Abstract base class for all agents."""
    
    role: AgentRole
    model_tier: str  # "lite", "flash", "flash-thinking", "pro"
    max_retries: int = 3
    timeout_seconds: int = 30
    
    def __init__(self, llm: Any): # utilizing Any to avoid cyclic dependency for now
        self.llm = llm
        self.execution_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass
    
    @abstractmethod
    def build_prompt(self, **kwargs) -> str:
        """Build the user prompt from inputs."""
        pass
    
    @abstractmethod
    def parse_response(self, response: str) -> T:
        """Parse LLM response into structured output."""
        pass
    
    async def execute(self, **kwargs) -> T:
        """Execute the agent task with retries."""
        self.execution_count += 1
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"{self.role.value}: Attempt {attempt + 1}")
                
                # Build prompt
                system = self.get_system_prompt()
                user_prompt = self.build_prompt(**kwargs)
                full_prompt = f"{system}\n\n{user_prompt}"
                
                # Call LLM
                result = await self.llm.generate(
                    prompt=full_prompt,
                    task=self._get_task_type(),
                    max_tokens=self._get_max_tokens(),
                )
                
                # Track usage
                self.total_tokens += result["tokens"]
                self.total_cost += result["cost"]
                
                # Parse response
                parsed = self.parse_response(result["text"])
                
                logger.info(f"{self.role.value}: Success")
                return parsed
                
            except Exception as e:
                logger.warning(f"{self.role.value}: Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
        
        raise RuntimeError(f"{self.role.value} failed after {self.max_retries} attempts")
    
    def _get_task_type(self) -> str:
        """Map agent role to LLM task type for routing."""
        mapping = {
            AgentRole.STRATEGIST: "reason",
            AgentRole.EVALUATOR: "evaluate",
            AgentRole.RESOLVER: "resolve",
            AgentRole.EXTRACTOR: "extract",
            AgentRole.GENERATOR: "generate",
            AgentRole.VALIDATOR: "validate",
            AgentRole.FORMATTER: "format",
        }
        return mapping.get(self.role, "generate")
    
    def _get_max_tokens(self) -> int:
        """Get max tokens based on model tier."""
        return {
            "lite": 500,
            "flash": 2000,
            "flash-thinking": 2000,
            "pro": 4000,
        }.get(self.model_tier, 1000)
    
    def get_stats(self) -> dict:
        """Return execution statistics."""
        return {
            "role": self.role.value,
            "executions": self.execution_count,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
        }
