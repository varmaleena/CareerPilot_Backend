import pytest
from unittest.mock import AsyncMock, patch

from app.agents.masters.strategist import StrategistAgent
from app.agents.masters.evaluator import EvaluatorAgent
from app.agents.masters.resolver import ResolverAgent
from app.models.agents import Verdict


class TestStrategistAgent:
    """Tests for Strategist agent."""
    
    @pytest.fixture
    def agent(self, mock_llm):
        return StrategistAgent(mock_llm)
    
    def test_agent_role(self, agent):
        """Test agent has correct role."""
        from app.models.agents import AgentRole
        assert agent.role == AgentRole.STRATEGIST
    
    def test_agent_model_tier(self, agent):
        """Test agent uses pro model tier."""
        assert agent.model_tier == "pro"
    
    def test_get_system_prompt(self, agent):
        """Test system prompt contains key instructions."""
        prompt = agent.get_system_prompt()
        assert "Strategist" in prompt
        assert "JSON" in prompt
    
    def test_build_prompt(self, agent):
        """Test prompt building."""
        prompt = agent.build_prompt(
            request_type="resume_analysis",
            user_input="Analyze my resume",
            context={"skills": ["Python", "AWS"]},
        )
        assert "resume_analysis" in prompt
        assert "Python" in prompt
    
    @pytest.mark.asyncio
    async def test_execute(self, agent, mock_llm):
        """Test agent execution."""
        mock_llm.generate.return_value = {
            "text": '''```json
{
    "complexity": "complex",
    "confidence": 85,
    "execution_plan": [
        {"step": 1, "agent": "extractor", "task": "Extract resume data", "timeout": 10, "fallback_if": "Empty resume"}
    ]
}
```''',
            "tokens": 100,
            "cost": 0.01,
        }
        
        result = await agent.execute(
            request_type="test",
            user_input="test input",
        )
        
        from app.models.agents import Complexity
        assert result.complexity == Complexity.COMPLEX
        assert result.confidence == 85
        assert len(result.execution_plan) == 1


class TestEvaluatorAgent:
    """Tests for Evaluator agent."""
    
    @pytest.fixture
    def agent(self, mock_llm):
        return EvaluatorAgent(mock_llm)
    
    def test_agent_role(self, agent):
        """Test agent has correct role."""
        from app.models.agents import AgentRole
        assert agent.role == AgentRole.EVALUATOR
    
    def test_agent_model_tier(self, agent):
        """Test agent uses flash-thinking model tier."""
        assert agent.model_tier == "flash-thinking"
    
    def test_get_system_prompt(self, agent):
        """Test system prompt contains evaluation instructions."""
        prompt = agent.get_system_prompt()
        assert "Evaluator" in prompt
        assert "score" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_execute(self, agent, mock_llm):
        """Test evaluator execution."""
        mock_llm.generate.return_value = {
            "text": '''```json
{
    "overall_score": 85,
    "verdict": "approve",
    "issues": ["Minor formatting issue"],
    "revision_instructions": null
}
```''',
            "tokens": 50,
            "cost": 0.005,
        }
        
        result = await agent.execute(
            output_type="resume",
            output_content="Sample resume content",
        )
        
        assert result.overall_score == 85
        assert result.verdict == Verdict.APPROVE
        assert len(result.issues) == 1


class TestResolverAgent:
    """Tests for Resolver agent."""
    
    @pytest.fixture
    def agent(self, mock_llm):
        return ResolverAgent(mock_llm)
    
    def test_agent_role(self, agent):
        """Test agent has correct role."""
        from app.models.agents import AgentRole
        assert agent.role == AgentRole.RESOLVER
    
    def test_agent_model_tier(self, agent):
        """Test agent uses pro model tier."""
        assert agent.model_tier == "pro"
    
    @pytest.mark.asyncio
    async def test_execute(self, agent, mock_llm):
        """Test resolver execution."""
        mock_llm.generate.return_value = {
            "text": '''```json
{
    "resolution": "Use approach A",
    "confidence": 90,
    "fallback_response": null,
    "should_notify_user": false,
    "user_message": null
}
```''',
            "tokens": 75,
            "cost": 0.008,
        }
        
        result = await agent.execute(
            issue_type="conflict",
            conflicting_outputs=[{"a": 1}, {"b": 2}],
        )
        
        assert result.resolution == "Use approach A"
        assert result.confidence == 90
        assert result.should_notify_user is False
