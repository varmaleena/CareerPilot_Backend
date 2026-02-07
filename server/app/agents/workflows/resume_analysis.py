from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from operator import add

from app.agents.masters.strategist import StrategistAgent
from app.agents.masters.evaluator import EvaluatorAgent
from app.agents.masters.resolver import ResolverAgent
from app.agents.helpers.extractor import ExtractorAgent
from app.agents.helpers.generator import GeneratorAgent
from app.agents.helpers.validator import ValidatorAgent
from app.agents.helpers.formatter import FormatterAgent
from app.services.llm.gateway import LLMGateway


class ResumeAnalysisState(TypedDict):
    # Input
    resume_text: str
    target_role: str
    target_company: str | None
    
    # Processing stages
    validation_result: dict | None
    extracted_data: dict | None
    analysis_result: dict | None
    evaluation_result: dict | None
    
    # Control
    retry_count: int
    current_stage: str
    errors: Annotated[list[str], add]
    
    # Output
    final_response: dict | None
    tokens_used: int
    cost_usd: float
    status: str


class ResumeAnalysisWorkflow:
    """LangGraph workflow for resume analysis."""
    
    def __init__(self, llm: LLMGateway):
        self.llm = llm
        
        # Initialize agents
        self.validator = ValidatorAgent(llm)
        self.extractor = ExtractorAgent(llm)
        self.strategist = StrategistAgent(llm)
        self.generator = GeneratorAgent(llm)
        self.evaluator = EvaluatorAgent(llm)
        self.resolver = ResolverAgent(llm)
        self.formatter = FormatterAgent(llm)
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        graph = StateGraph(ResumeAnalysisState)
        
        # Add nodes
        graph.add_node("validate", self._validate_node)
        graph.add_node("extract", self._extract_node)
        graph.add_node("analyze", self._analyze_node)
        graph.add_node("evaluate", self._evaluate_node)
        graph.add_node("format", self._format_node)
        graph.add_node("resolve", self._resolve_node)
        
        # Add edges
        graph.set_entry_point("validate")
        
        graph.add_conditional_edges(
            "validate",
            self._route_after_validation,
            {
                "extract": "extract",
                "resolve": "resolve",
            }
        )
        
        graph.add_conditional_edges(
            "extract",
            self._route_after_extraction,
            {
                "analyze": "analyze",
                "extract": "extract",  # Retry
                "resolve": "resolve",
            }
        )
        
        graph.add_edge("analyze", "evaluate")
        
        graph.add_conditional_edges(
            "evaluate",
            self._route_after_evaluation,
            {
                "format": "format",
                "analyze": "analyze",  # Revise
                "resolve": "resolve",
            }
        )
        
        graph.add_edge("format", END)
        graph.add_edge("resolve", END)
        
        return graph.compile()
    
    # Node implementations
    async def _validate_node(self, state: ResumeAnalysisState) -> dict:
        """Validate the resume input."""
        result = await self.validator.execute(
            content=state["resume_text"],
            expected_type="resume",
        )
        
        return {
            "validation_result": result.model_dump(),
            "current_stage": "validated",
            "tokens_used": state["tokens_used"] + self.validator.total_tokens,
            "cost_usd": state["cost_usd"] + self.validator.total_cost,
        }
    
    async def _extract_node(self, state: ResumeAnalysisState) -> dict:
        """Extract structured data from resume."""
        result = await self.extractor.execute(
            document_text=state["resume_text"],
        )
        
        return {
            "extracted_data": result.model_dump(),
            "current_stage": "extracted",
            "tokens_used": state["tokens_used"] + self.extractor.total_tokens,
            "cost_usd": state["cost_usd"] + self.extractor.total_cost,
        }
    
    async def _analyze_node(self, state: ResumeAnalysisState) -> dict:
        """Generate analysis based on extracted data."""
        result = await self.generator.execute(
            generation_type="resume_analysis",
            context={
                "extracted_data": state["extracted_data"],
                "target_role": state["target_role"],
                "target_company": state.get("target_company"),
            },
            requirements=[
                "Assess readiness score (0-100)",
                "Identify skill gaps",
                "List strengths and weaknesses",
                "Provide actionable recommendations",
                "Calculate ATS compatibility score",
            ],
        )
        
        return {
            "analysis_result": result.model_dump(),
            "current_stage": "analyzed",
            "tokens_used": state["tokens_used"] + self.generator.total_tokens,
            "cost_usd": state["cost_usd"] + self.generator.total_cost,
        }
    
    async def _evaluate_node(self, state: ResumeAnalysisState) -> dict:
        """Evaluate analysis quality."""
        result = await self.evaluator.execute(
            output_type="resume_analysis",
            output_content=str(state["analysis_result"]),
            rubric="""
            - Readiness score is justified (20 pts)
            - Skill gaps are specific and actionable (25 pts)
            - Recommendations are prioritized (25 pts)
            - Analysis is tailored to target role (20 pts)
            - Output is well-structured (10 pts)
            """,
        )
        
        return {
            "evaluation_result": result.model_dump(),
            "current_stage": "evaluated",
            "tokens_used": state["tokens_used"] + self.evaluator.total_tokens,
            "cost_usd": state["cost_usd"] + self.evaluator.total_cost,
        }
    
    async def _format_node(self, state: ResumeAnalysisState) -> dict:
        """Format final output."""
        result = await self.formatter.execute(
            content=str(state["analysis_result"]),
            target_format="json",
        )
        
        return {
            "final_response": {
                "analysis": state["analysis_result"],
                "formatted": result.formatted_content,
            },
            "status": "completed",
            "tokens_used": state["tokens_used"] + self.formatter.total_tokens,
            "cost_usd": state["cost_usd"] + self.formatter.total_cost,
        }
    
    async def _resolve_node(self, state: ResumeAnalysisState) -> dict:
        """Handle failures gracefully."""
        result = await self.resolver.execute(
            issue_type="workflow_failure",
            conflicting_outputs=[
                {"stage": state["current_stage"], "errors": state["errors"]}
            ],
            context={"resume_preview": state["resume_text"][:500]},
        )
        
        return {
            "final_response": {
                "fallback": result.fallback_response or result.resolution,
                "confidence": result.confidence,
            },
            "status": "completed_with_fallback",
        }
    
    # Routing functions
    def _route_after_validation(self, state: ResumeAnalysisState) -> str:
        validation = state.get("validation_result", {})
        if validation.get("is_valid") and validation.get("confidence", 0) >= 50:
            return "extract"
        return "resolve"
    
    def _route_after_extraction(self, state: ResumeAnalysisState) -> str:
        if state.get("extracted_data"):
            return "analyze"
        if state.get("retry_count", 0) < 2:
            return "extract"
        return "resolve"
    
    def _route_after_evaluation(self, state: ResumeAnalysisState) -> str:
        evaluation = state.get("evaluation_result", {})
        score = evaluation.get("overall_score", 0)
        verdict = evaluation.get("verdict", "reject")
        
        if verdict == "approve" or score >= 80:
            return "format"
        if verdict == "revise" and state.get("retry_count", 0) < 2:
            return "analyze"
        return "resolve"
    
    # Public interface
    async def run(
        self,
        resume_text: str,
        target_role: str,
        target_company: str | None = None,
    ) -> dict:
        """Execute the resume analysis workflow."""
        initial_state = {
            "resume_text": resume_text,
            "target_role": target_role,
            "target_company": target_company,
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
        
        result = await self.graph.ainvoke(initial_state)
        return result
