from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    model: str
    max_tokens: int
    temperature: float
    cost_per_1k_tokens: float


class ModelRouter:
    """Cost-aware model selection."""
    
    MODELS = {
        "lite": ModelConfig(
            model="gemini-2.0-flash-lite",
            max_tokens=500,
            temperature=0.3,
            cost_per_1k_tokens=0.0001,
        ),
        "flash": ModelConfig(
            model="gemini-2.5-flash",
            max_tokens=2000,
            temperature=0.7,
            cost_per_1k_tokens=0.001,
        ),
        "flash-thinking": ModelConfig(
            model="gemini-2.5-flash-thinking",
            max_tokens=2000,
            temperature=0.5,
            cost_per_1k_tokens=0.002,
        ),
        "pro": ModelConfig(
            model="gemini-2.5-pro",
            max_tokens=4000,
            temperature=0.7,
            cost_per_1k_tokens=0.005,
        ),
    }
    
    TASK_ROUTING = {
        "validate": "lite",
        "extract": "lite",
        "format": "lite",
        "generate": "flash",
        "evaluate": "flash-thinking",
        "reason": "pro",
        "resolve": "pro",
    }
    
    def route(
        self, task: Literal["validate", "extract", "generate", "evaluate", "reason", "resolve", "format"]
    ) -> ModelConfig:
        """Get optimal model config for task."""
        model_key = self.TASK_ROUTING.get(task, "flash")
        return self.MODELS[model_key]
