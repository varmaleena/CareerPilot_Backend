import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestResumeAnalysisWorkflow:
    """Tests for Resume Analysis Workflow."""
    
    @pytest.mark.asyncio
    async def test_workflow_initialization(self, mock_llm):
        """Test workflow can be initialized."""
        from app.agents.workflows.resume_analysis import ResumeAnalysisWorkflow
        
        workflow = ResumeAnalysisWorkflow(mock_llm)
        assert workflow.graph is not None
    
    @pytest.mark.asyncio
    async def test_workflow_state_structure(self):
        """Test initial state has all required fields."""
        from app.agents.workflows.resume_analysis import ResumeAnalysisState
        
        state: ResumeAnalysisState = {
            "resume_text": "test",
            "target_role": "SWE",
            "target_company": None,
            "validation_result": None,
            "extracted_data": None,
            "analysis_result": None,
            "evaluation_result": None,
            "retry_count": 0,
            "current_stage": "started",
            "errors": [],
            "final_response": None,
            "tokens_used": 0,
            "cost_usd": 0.0,
            "status": "pending",
        }
        
        assert state["current_stage"] == "started"
        assert state["tokens_used"] == 0


class TestInterviewWorkflow:
    """Tests for Interview Workflow."""
    
    @pytest.mark.asyncio
    async def test_workflow_initialization(self, mock_llm):
        """Test interview workflow can be initialized."""
        from app.agents.workflows.interview import InterviewWorkflow
        
        workflow = InterviewWorkflow(mock_llm)
        assert workflow.graph is not None
    
    def test_interview_type_enum(self):
        """Test interview types are defined."""
        from app.agents.workflows.interview import InterviewType
        
        assert InterviewType.BEHAVIORAL.value == "behavioral"
        assert InterviewType.TECHNICAL.value == "technical"
        assert InterviewType.SYSTEM_DESIGN.value == "system_design"


class TestLearningPlanWorkflow:
    """Tests for Learning Plan Workflow."""
    
    @pytest.mark.asyncio
    async def test_workflow_initialization(self, mock_llm):
        """Test learning plan workflow can be initialized."""
        from app.agents.workflows.learning_plan import LearningPlanWorkflow
        
        workflow = LearningPlanWorkflow(mock_llm)
        assert workflow.graph is not None
    
    @pytest.mark.asyncio
    async def test_state_structure(self):
        """Test learning plan state structure."""
        from app.agents.workflows.learning_plan import LearningPlanState
        
        state: LearningPlanState = {
            "user_id": "user-123",
            "target_role": "Data Scientist",
            "current_skills": ["Python", "SQL"],
            "skill_gaps": [{"skill": "ML", "priority": "high"}],
            "available_hours_per_week": 10,
            "deadline_weeks": 12,
            "learning_style": "mixed",
            "prioritized_skills": None,
            "resource_recommendations": None,
            "milestone_plan": None,
            "project_ideas": None,
            "current_stage": "started",
            "retry_count": 0,
            "errors": [],
            "final_plan": None,
            "tokens_used": 0,
            "cost_usd": 0.0,
            "status": "pending",
        }
        
        assert state["target_role"] == "Data Scientist"


class TestResumeOptimizationWorkflow:
    """Tests for Resume Optimization Workflow."""
    
    @pytest.mark.asyncio
    async def test_workflow_initialization(self, mock_llm):
        """Test resume optimization workflow can be initialized."""
        from app.agents.workflows.resume_optimization import ResumeOptimizationWorkflow
        
        workflow = ResumeOptimizationWorkflow(mock_llm)
        assert workflow.graph is not None
    
    @pytest.mark.asyncio
    async def test_state_includes_ats_scores(self):
        """Test state tracks ATS scores before and after."""
        from app.agents.workflows.resume_optimization import ResumeOptimizationState
        
        state: ResumeOptimizationState = {
            "resume_text": "test",
            "job_description": "job",
            "target_role": "SWE",
            "optimization_focus": ["ats", "bullets"],
            "current_resume_data": None,
            "job_requirements": None,
            "keyword_analysis": None,
            "ats_score_before": None,
            "bullet_rewrites": None,
            "suggested_additions": None,
            "optimized_resume": None,
            "ats_score_after": None,
            "changes_made": None,
            "final_report": None,
            "current_stage": "started",
            "retry_count": 0,
            "errors": [],
            "tokens_used": 0,
            "cost_usd": 0.0,
            "status": "pending",
        }
        
        assert "ats_score_before" in state
        assert "ats_score_after" in state
