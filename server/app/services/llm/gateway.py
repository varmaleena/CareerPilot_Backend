from typing import Literal
import google.generativeai as genai
from app.services.llm.key_manager import KeyManager
from app.services.llm.model_router import ModelRouter
from app.services.llm.token_counter import TokenCounter
from app.services.cache.semantic_cache import SemanticCache


class LLMGateway:
    """Unified LLM interface with cost optimization."""
    
    def __init__(self):
        self.key_manager = KeyManager()
        self.model_router = ModelRouter()
        self.token_counter = TokenCounter()
        self.cache = SemanticCache()
    
    async def generate(
        self,
        prompt: str,
        task: Literal["validate", "extract", "generate", "reason", "evaluate", "resolve", "format"],
        max_tokens: int | None = None,
        use_cache: bool = True,
    ) -> dict:
        """Generate LLM response with optimizations."""
        
        # Try cache first
        if use_cache:
            cached = await self.cache.get(prompt, task)
            if cached:
                return {"text": cached, "cached": True, "tokens": 0, "cost": 0}
        
        # Get optimal model and key
        model_config = self.model_router.route(task)
        api_key = self.key_manager.get_next_key()
        
        # Configure client
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_config.model)
        
        # Generate
        response = await model.generate_content_async(
            prompt,
            generation_config={
                "max_output_tokens": max_tokens or model_config.max_tokens,
                "temperature": model_config.temperature,
            }
        )
        
        text = response.text
        
        # Calculate tokens and cost
        tokens = self.token_counter.count(prompt, text)
        cost = self.token_counter.calculate_cost(tokens, model_config.model)
        
        # Cache result
        if use_cache:
            await self.cache.set(prompt, text, task)
        
        return {
            "text": text,
            "cached": False,
            "tokens": tokens,
            "cost": cost,
            "model": model_config.model,
        }
