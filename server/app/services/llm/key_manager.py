import asyncio
from app.config import settings


class KeyManager:
    """Rotate Gemini API keys to avoid rate limits."""
    
    def __init__(self):
        self.keys = settings.gemini_keys_list
        self.current_index = 0
        self.lock = asyncio.Lock()
        self.failed_keys: set[int] = set()
    
    def get_next_key(self) -> str:
        """Get next available API key (round-robin)."""
        # Skip failed keys
        attempts = 0
        while attempts < len(self.keys):
            key_index = self.current_index % len(self.keys)
            self.current_index += 1
            
            if key_index not in self.failed_keys:
                return self.keys[key_index]
            attempts += 1
        
        # All keys failed, reset and try first
        self.failed_keys.clear()
        return self.keys[0]
    
    def mark_failed(self, key: str) -> None:
        """Mark key as temporarily failed."""
        try:
            index = self.keys.index(key)
            self.failed_keys.add(index)
        except ValueError:
            pass
    
    def reset_failed(self) -> None:
        """Reset all failed keys."""
        self.failed_keys.clear()
