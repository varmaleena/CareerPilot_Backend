class TokenCounter:
    """Count tokens and calculate costs."""
    
    MODEL_COSTS = {
        "gemini-2.0-flash-lite": 0.0001,
        "gemini-2.5-flash": 0.001,
        "gemini-2.5-flash-thinking": 0.002,
        "gemini-2.5-pro": 0.005,
    }
    
    def count(self, prompt: str, response: str) -> int:
        """Estimate token count (rough approximation)."""
        # Rough estimation: 1 token ≈ 4 characters
        total_chars = len(prompt) + len(response)
        return total_chars // 4
    
    def calculate_cost(self, tokens: int, model: str) -> float:
        """Calculate cost in USD."""
        cost_per_1k = self.MODEL_COSTS.get(model, 0.001)
        return (tokens / 1000) * cost_per_1k
