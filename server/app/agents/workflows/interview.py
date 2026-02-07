from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Literal
from operator import add
from enum import Enum

from app.agents.masters.strategist import StrategistAgent
from app.agents.masters.evaluator import EvaluatorAgent
from app.agents.helpers.generator import GeneratorAgent
from app.agents.helpers.validator import ValidatorAgent
from app.services.llm.gateway import LLMGateway


class InterviewType(str, Enum):
    BEHAVIORAL = "behavioral"
    TECHNICAL = "technical"
    SYSTEM_DESIGN = "system_design"
    CASE_STUDY = "case_study"


class InterviewState(TypedDict):
    # Session config
    session_id: str
    interview_type: InterviewType
    target_role: str
    difficulty: Literal["easy", "medium", "hard"]
    
    # Conversation
    messages: list[dict]  # [{role: "interviewer"|"candidate", content: str}]
    current_question: str | None
    question_count: int
    max_questions: int
    
    # Candidate context
    resume_summary: dict | None
    
    # Evaluation
    answer_scores: list[dict]  # [{question_id, score, feedback}]
    overall_score: int | None
    
    # Control
    current_stage: str
    errors: Annotated[list[str], add]
    
    # Output
    final_feedback: dict | None
    tokens_used: int
    cost_usd: float
    status: str


class InterviewWorkflow:
    """LangGraph workflow for mock interviews with real-time feedback."""
    
    def __init__(self, llm: LLMGateway):
        self.llm = llm
        
        self.strategist = StrategistAgent(llm)
        self.generator = GeneratorAgent(llm)
        self.evaluator = EvaluatorAgent(llm)
        self.validator = ValidatorAgent(llm)
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        graph = StateGraph(InterviewState)
        
        # Nodes
        graph.add_node("generate_question", self._generate_question_node)
        graph.add_node("await_answer", self._await_answer_node)
        graph.add_node("evaluate_answer", self._evaluate_answer_node)
        graph.add_node("generate_feedback", self._generate_feedback_node)
        graph.add_node("finalize", self._finalize_node)
        
        # Flow
        graph.set_entry_point("generate_question")
        
        graph.add_edge("generate_question", "await_answer")
        graph.add_edge("await_answer", "evaluate_answer")
        
        graph.add_conditional_edges(
            "evaluate_answer",
            self._route_after_evaluation,
            {
                "next_question": "generate_feedback",
                "finalize": "finalize",
            }
        )
        
        graph.add_conditional_edges(
            "generate_feedback",
            self._route_after_feedback,
            {
                "continue": "generate_question",
                "end": "finalize",
            }
        )
        
        graph.add_edge("finalize", END)
        
        return graph.compile()
    
    async def _generate_question_node(self, state: InterviewState) -> dict:
        """Generate the next interview question based on context."""
        result = await self.generator.execute(
            generation_type="interview_question",
            context={
                "interview_type": state["interview_type"],
                "target_role": state["target_role"],
                "difficulty": state["difficulty"],
                "previous_questions": [m["content"] for m in state["messages"] if m.get("role") == "interviewer"],
                "question_number": state["question_count"] + 1,
                "resume_summary": state.get("resume_summary"),
            },
            requirements=[
                f"Generate a {state['difficulty']} difficulty question",
                f"Focus on {state['interview_type']} skills",
                "Build on previous questions if applicable",
                "Be specific and practical",
            ],
        )
        
        question = result.content
        
        return {
            "current_question": question,
            "question_count": state["question_count"] + 1,
            "messages": state["messages"] + [{"role": "interviewer", "content": question}],
            "current_stage": "awaiting_answer",
            "tokens_used": state["tokens_used"] + self.generator.total_tokens,
            "cost_usd": state["cost_usd"] + self.generator.total_cost,
        }
    
    async def _await_answer_node(self, state: InterviewState) -> dict:
        """Placeholder - answer is injected via WebSocket."""
        return {"current_stage": "answer_received"}
    
    async def _evaluate_answer_node(self, state: InterviewState) -> dict:
        """Evaluate the candidate's answer."""
        candidate_answer = None
        for msg in reversed(state["messages"]):
            if msg.get("role") == "candidate":
                candidate_answer = msg["content"]
                break
        
        if not candidate_answer:
            return {"errors": state["errors"] + ["No candidate answer found"]}
        
        result = await self.evaluator.execute(
            output_type="interview_answer",
            output_content=f"""
Question: {state['current_question']}

Candidate Answer: {candidate_answer}
""",
            rubric=self._get_evaluation_rubric(state["interview_type"]),
        )
        
        score_entry = {
            "question_id": state["question_count"],
            "question": state["current_question"],
            "answer": candidate_answer,
            "score": result.overall_score,
            "issues": result.issues,
            "verdict": result.verdict.value,
        }
        
        return {
            "answer_scores": state["answer_scores"] + [score_entry],
            "current_stage": "evaluated",
            "tokens_used": state["tokens_used"] + self.evaluator.total_tokens,
            "cost_usd": state["cost_usd"] + self.evaluator.total_cost,
        }
    
    async def _generate_feedback_node(self, state: InterviewState) -> dict:
        """Generate real-time feedback for the candidate."""
        latest_score = state["answer_scores"][-1]
        
        result = await self.generator.execute(
            generation_type="interview_feedback",
            context={
                "question": latest_score["question"],
                "answer": latest_score["answer"],
                "score": latest_score["score"],
                "issues": latest_score["issues"],
            },
            requirements=[
                "Be constructive and encouraging",
                "Provide specific improvement suggestions",
                "Keep feedback concise (2-3 sentences)",
            ],
        )
        
        feedback_message = {"role": "interviewer", "content": f"📝 Feedback: {result.content}"}
        
        return {
            "messages": state["messages"] + [feedback_message],
            "current_stage": "feedback_delivered",
            "tokens_used": state["tokens_used"] + self.generator.total_tokens,
            "cost_usd": state["cost_usd"] + self.generator.total_cost,
        }
    
    async def _finalize_node(self, state: InterviewState) -> dict:
        """Generate final interview summary and recommendations."""
        avg_score = sum(s["score"] for s in state["answer_scores"]) / len(state["answer_scores"])
        
        result = await self.generator.execute(
            generation_type="interview_summary",
            context={
                "interview_type": state["interview_type"],
                "target_role": state["target_role"],
                "scores": state["answer_scores"],
                "average_score": avg_score,
            },
            requirements=[
                "Summarize overall performance",
                "Highlight strengths and weaknesses",
                "Provide 3-5 actionable improvement areas",
                "Give hiring recommendation (Strong Yes / Yes / Maybe / No)",
            ],
        )
        
        return {
            "overall_score": int(avg_score),
            "final_feedback": {
                "summary": result.content,
                "average_score": avg_score,
                "question_count": len(state["answer_scores"]),
                "key_points": result.key_points,
            },
            "status": "completed",
            "tokens_used": state["tokens_used"] + self.generator.total_tokens,
            "cost_usd": state["cost_usd"] + self.generator.total_cost,
        }
    
    def _route_after_evaluation(self, state: InterviewState) -> str:
        if state["question_count"] >= state["max_questions"]:
            return "finalize"
        return "next_question"
    
    def _route_after_feedback(self, state: InterviewState) -> str:
        if state["question_count"] >= state["max_questions"]:
            return "end"
        return "continue"
    
    def _get_evaluation_rubric(self, interview_type: InterviewType) -> str:
        rubrics = {
            InterviewType.BEHAVIORAL: """
- STAR format usage (20 pts)
- Specific examples provided (25 pts)
- Relevance to question (25 pts)
- Communication clarity (15 pts)
- Self-awareness demonstrated (15 pts)
""",
            InterviewType.TECHNICAL: """
- Technical accuracy (30 pts)
- Problem-solving approach (25 pts)
- Code quality discussed (20 pts)
- Trade-offs considered (15 pts)
- Communication of thought process (10 pts)
""",
            InterviewType.SYSTEM_DESIGN: """
- Requirements clarification (15 pts)
- High-level design (25 pts)
- Component deep-dive (20 pts)
- Scalability considerations (20 pts)
- Trade-off discussions (20 pts)
""",
            InterviewType.CASE_STUDY: """
- Problem understanding (20 pts)
- Structured approach (25 pts)
- Quantitative analysis (20 pts)
- Creative solutions (20 pts)
- Recommendation clarity (15 pts)
""",
        }
        return rubrics.get(interview_type, rubrics[InterviewType.BEHAVIORAL])
    
    async def add_candidate_answer(self, state: dict, answer: str) -> dict:
        """Add candidate answer and continue workflow."""
        updated_state = {
            **state,
            "messages": state["messages"] + [{"role": "candidate", "content": answer}],
        }
        return await self.graph.ainvoke(updated_state)
    
    async def start(
        self,
        session_id: str,
        interview_type: InterviewType,
        target_role: str,
        difficulty: str = "medium",
        max_questions: int = 5,
        resume_summary: dict | None = None,
    ) -> dict:
        """Start a new interview session."""
        initial_state = {
            "session_id": session_id,
            "interview_type": interview_type,
            "target_role": target_role,
            "difficulty": difficulty,
            "messages": [],
            "current_question": None,
            "question_count": 0,
            "max_questions": max_questions,
            "resume_summary": resume_summary,
            "answer_scores": [],
            "overall_score": None,
            "current_stage": "started",
            "errors": [],
            "final_feedback": None,
            "tokens_used": 0,
            "cost_usd": 0.0,
            "status": "in_progress",
        }
        
        result = await self.graph.ainvoke(initial_state)
        return result
