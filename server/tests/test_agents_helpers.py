import pytest
from unittest.mock import AsyncMock

from app.agents.helpers.extractor import ExtractorAgent
from app.agents.helpers.generator import GeneratorAgent
from app.agents.helpers.validator import ValidatorAgent
from app.agents.helpers.formatter import FormatterAgent


class TestExtractorAgent:
    """Tests for Extractor agent."""
    
    @pytest.fixture
    def agent(self, mock_llm):
        return ExtractorAgent(mock_llm)
    
    def test_agent_uses_lite_model(self, agent):
        """Extractor should use cheap lite model."""
        assert agent.model_tier == "lite"
    
    def test_get_system_prompt(self, agent):
        """Test system prompt."""
        prompt = agent.get_system_prompt()
        assert "Extractor" in prompt
        assert "JSON" in prompt
    
    @pytest.mark.asyncio
    async def test_extract_resume(self, agent, mock_llm):
        """Test resume extraction."""
        mock_llm.generate.return_value = {
            "text": '''```json
{
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "555-1234",
    "summary": "Experienced developer",
    "skills": ["Python", "JavaScript"],
    "experience": [],
    "education": [],
    "certifications": [],
    "projects": []
}
```''',
            "tokens": 80,
            "cost": 0.001,
        }
        
        result = await agent.execute(document_text="John Doe, john@example.com...")
        
        assert result.name == "John Doe"
        assert result.email == "john@example.com"
        assert "Python" in result.skills


class TestGeneratorAgent:
    """Tests for Generator agent."""
    
    @pytest.fixture
    def agent(self, mock_llm):
        return GeneratorAgent(mock_llm)
    
    def test_agent_uses_flash_model(self, agent):
        """Generator should use flash model for speed/quality balance."""
        assert agent.model_tier == "flash"
    
    @pytest.mark.asyncio
    async def test_generate_content(self, agent, mock_llm):
        """Test content generation."""
        mock_llm.generate.return_value = {
            "text": '''```json
{
    "content": "Here is your analysis...",
    "content_type": "analysis",
    "word_count": 150,
    "key_points": ["Strong skills", "Good experience"]
}
```''',
            "tokens": 100,
            "cost": 0.002,
        }
        
        result = await agent.execute(
            generation_type="resume_analysis",
            context={"skills": ["Python"]},
            requirements=["Be specific"],
        )
        
        assert "analysis" in result.content
        assert result.content_type == "analysis"
        assert len(result.key_points) == 2


class TestValidatorAgent:
    """Tests for Validator agent."""
    
    @pytest.fixture
    def agent(self, mock_llm):
        return ValidatorAgent(mock_llm)
    
    def test_agent_uses_lite_model(self, agent):
        """Validator should use lite model for quick validation."""
        assert agent.model_tier == "lite"
    
    def test_short_timeout(self, agent):
        """Validator should have short timeout."""
        assert agent.timeout_seconds <= 10
    
    @pytest.mark.asyncio
    async def test_validate_document(self, agent, mock_llm):
        """Test document validation."""
        mock_llm.generate.return_value = {
            "text": '''```json
{
    "is_valid": true,
    "document_type": "resume",
    "confidence": 95,
    "issues": [],
    "suggestions": ["Add more metrics"]
}
```''',
            "tokens": 40,
            "cost": 0.0005,
        }
        
        result = await agent.execute(
            content="John Doe, Software Engineer...",
            expected_type="resume",
        )
        
        assert result.is_valid is True
        assert result.document_type == "resume"
        assert result.confidence == 95


class TestFormatterAgent:
    """Tests for Formatter agent."""
    
    @pytest.fixture
    def agent(self, mock_llm):
        return FormatterAgent(mock_llm)
    
    def test_agent_uses_lite_model(self, agent):
        """Formatter should use lite model."""
        assert agent.model_tier == "lite"
    
    @pytest.mark.asyncio
    async def test_format_content(self, agent, mock_llm):
        """Test content formatting."""
        mock_llm.generate.return_value = {
            "text": '''```json
{
    "formatted_content": "# Resume\\n\\n**John Doe**\\n",
    "format_type": "markdown",
    "character_count": 30
}
```''',
            "tokens": 50,
            "cost": 0.0005,
        }
        
        result = await agent.execute(
            content="John Doe resume data",
            target_format="markdown",
        )
        
        assert "John Doe" in result.formatted_content
        assert result.format_type == "markdown"
