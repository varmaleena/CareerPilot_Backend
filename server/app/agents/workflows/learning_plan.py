from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Literal
from operator import add
from datetime import datetime, timedelta

from app.agents.masters.strategist import StrategistAgent
from app.agents.masters.evaluator import EvaluatorAgent
from app.agents.helpers.generator import GeneratorAgent
from app.agents.helpers.extractor import ExtractorAgent
from app.services.llm.gateway import LLMGateway


class LearningPlanState(TypedDict):
    # Input
    user_id: str
    target_role: str
    current_skills: list[str]
    skill_gaps: list[dict]
    available_hours_per_week: int
    deadline_weeks: int | None
    learning_style: Literal["video", "reading", "hands-on", "mixed"]
    
    # Processing
    prioritized_skills: list[dict] | None
    resource_recommendations: list[dict] | None
    milestone_plan: list[dict] | None
    project_ideas: list[dict] | None
    
    # Control
    current_stage: str
    retry_count: int
    errors: Annotated[list[str], add]
    
    # Output
    final_plan: dict | None
    tokens_used: int
    cost_usd: float
    status: str


class LearningPlanWorkflow:
    """LangGraph workflow for personalized learning plan generation."""
    
    def __init__(self, llm: LLMGateway):
        self.llm = llm
        
        self.strategist = StrategistAgent(llm)
        self.generator = GeneratorAgent(llm)
        self.evaluator = EvaluatorAgent(llm)
        self.extractor = ExtractorAgent(llm)
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        graph = StateGraph(LearningPlanState)
        
        # Nodes
        graph.add_node("prioritize", self._prioritize_node)
        graph.add_node("recommend_resources", self._recommend_resources_node)
        graph.add_node("create_milestones", self._create_milestones_node)
        graph.add_node("suggest_projects", self._suggest_projects_node)
        graph.add_node("evaluate", self._evaluate_node)
        graph.add_node("compile", self._compile_node)
        
        # Flow
        graph.set_entry_point("prioritize")
        
        graph.add_edge("prioritize", "recommend_resources")
        graph.add_edge("recommend_resources", "create_milestones")
        graph.add_edge("create_milestones", "suggest_projects")
        graph.add_edge("suggest_projects", "evaluate")
        
        graph.add_conditional_edges(
            "evaluate",
            self._route_after_evaluation,
            {
                "compile": "compile",
                "revise": "prioritize",
            }
        )
        
        graph.add_edge("compile", END)
        
        return graph.compile()
    
    async def _prioritize_node(self, state: LearningPlanState) -> dict:
        """Prioritize skills based on impact and learning curve."""
        result = await self.strategist.execute(
            request_type="skill_prioritization",
            user_input=f"""
Target Role: {state['target_role']}
Current Skills: {', '.join(state['current_skills'])}
Skill Gaps: {state['skill_gaps']}
Available Time: {state['available_hours_per_week']} hours/week
Deadline: {state.get('deadline_weeks', 'flexible')} weeks
""",
            context={
                "goal": "Prioritize skills by impact on getting the target role",
                "constraints": ["Time available", "Prerequisite dependencies"],
            },
        )
        
        prioritized = []
        for i, step in enumerate(result.execution_plan):
            prioritized.append({
                "skill": step.task,
                "priority": i + 1,
                "estimated_weeks": 2,
                "reason": step.fallback_if or "High impact for target role",
            })
        
        return {
            "prioritized_skills": prioritized,
            "current_stage": "prioritized",
            "tokens_used": state["tokens_used"] + self.strategist.total_tokens,
            "cost_usd": state["cost_usd"] + self.strategist.total_cost,
        }
    
    async def _recommend_resources_node(self, state: LearningPlanState) -> dict:
        """Recommend learning resources for each skill."""
        result = await self.generator.execute(
            generation_type="learning_resources",
            context={
                "skills": state["prioritized_skills"],
                "learning_style": state["learning_style"],
                "target_role": state["target_role"],
            },
            requirements=[
                "Include free and paid options",
                "Mix of video courses, articles, and hands-on tutorials",
                "Prioritize up-to-date resources (2023+)",
                "Include estimated completion time for each",
                "Add difficulty level (beginner/intermediate/advanced)",
            ],
        )
        
        resources = []
        for skill in state["prioritized_skills"]:
            resources.append({
                "skill": skill["skill"],
                "resources": [
                    {"type": "course", "name": f"Udemy: {skill['skill']} Masterclass", "url": "#", "hours": 15, "cost": "paid"},
                    {"type": "docs", "name": "Official Documentation", "url": "#", "hours": 5, "cost": "free"},
                    {"type": "practice", "name": "LeetCode/Project-based", "url": "#", "hours": 10, "cost": "free"},
                ],
            })
        
        return {
            "resource_recommendations": resources,
            "current_stage": "resources_recommended",
            "tokens_used": state["tokens_used"] + self.generator.total_tokens,
            "cost_usd": state["cost_usd"] + self.generator.total_cost,
        }
    
    async def _create_milestones_node(self, state: LearningPlanState) -> dict:
        """Create weekly milestones and checkpoints."""
        hours_per_week = state["available_hours_per_week"]
        deadline_weeks = state.get("deadline_weeks") or 12
        
        result = await self.generator.execute(
            generation_type="learning_milestones",
            context={
                "skills": state["prioritized_skills"],
                "resources": state["resource_recommendations"],
                "hours_per_week": hours_per_week,
                "total_weeks": deadline_weeks,
            },
            requirements=[
                "Create weekly learning targets",
                "Include measurable outcomes for each week",
                "Add checkpoint quizzes/assessments",
                "Build in buffer time for review",
                "Ensure realistic pace",
            ],
        )
        
        milestones = []
        weeks_per_skill = max(1, deadline_weeks // len(state["prioritized_skills"]))
        
        start_date = datetime.now()
        for i, skill in enumerate(state["prioritized_skills"]):
            milestone = {
                "skill": skill["skill"],
                "week_start": i * weeks_per_skill + 1,
                "week_end": (i + 1) * weeks_per_skill,
                "start_date": (start_date + timedelta(weeks=i * weeks_per_skill)).isoformat(),
                "goals": [
                    f"Complete foundational learning for {skill['skill']}",
                    f"Build a mini-project using {skill['skill']}",
                    "Pass checkpoint assessment",
                ],
                "hours_allocated": hours_per_week * weeks_per_skill,
                "checkpoint": f"Week {(i + 1) * weeks_per_skill} Assessment",
            }
            milestones.append(milestone)
        
        return {
            "milestone_plan": milestones,
            "current_stage": "milestones_created",
            "tokens_used": state["tokens_used"] + self.generator.total_tokens,
            "cost_usd": state["cost_usd"] + self.generator.total_cost,
        }
    
    async def _suggest_projects_node(self, state: LearningPlanState) -> dict:
        """Suggest portfolio projects that demonstrate learned skills."""
        result = await self.generator.execute(
            generation_type="portfolio_projects",
            context={
                "skills": [s["skill"] for s in state["prioritized_skills"]],
                "target_role": state["target_role"],
            },
            requirements=[
                "Suggest 3-5 portfolio-worthy projects",
                "Each project should combine multiple skills",
                "Include difficulty estimate and time required",
                "Make projects relevant to target role",
                "Include stretch goals for each project",
            ],
        )
        
        projects = [
            {
                "name": "Full-Stack Portfolio App",
                "description": "A personal portfolio with blog and project showcase",
                "skills_demonstrated": state["prioritized_skills"][:2],
                "difficulty": "intermediate",
                "estimated_hours": 40,
                "stretch_goals": ["Add CMS", "Implement dark mode", "Add animations"],
            },
            {
                "name": f"{state['target_role']} Case Study",
                "description": f"Real-world project simulating {state['target_role']} responsibilities",
                "skills_demonstrated": state["prioritized_skills"],
                "difficulty": "advanced",
                "estimated_hours": 60,
                "stretch_goals": ["Add documentation", "Deploy to production"],
            },
        ]
        
        return {
            "project_ideas": projects,
            "current_stage": "projects_suggested",
            "tokens_used": state["tokens_used"] + self.generator.total_tokens,
            "cost_usd": state["cost_usd"] + self.generator.total_cost,
        }
    
    async def _evaluate_node(self, state: LearningPlanState) -> dict:
        """Evaluate the completeness and quality of the learning plan."""
        plan_summary = f"""
Prioritized Skills: {len(state['prioritized_skills'])}
Resources per Skill: {len(state['resource_recommendations'][0]['resources']) if state['resource_recommendations'] else 0}
Milestones: {len(state['milestone_plan'])}
Project Ideas: {len(state['project_ideas'])}
Timeline: {state.get('deadline_weeks', 12)} weeks
Hours/Week: {state['available_hours_per_week']}
"""
        
        result = await self.evaluator.execute(
            output_type="learning_plan",
            output_content=plan_summary,
            rubric="""
- Realistic timeline (25 pts)
- Resource quality and variety (20 pts)
- Clear milestones with measurable outcomes (25 pts)
- Portfolio projects aligned with goals (20 pts)
- Personalization to user context (10 pts)
""",
        )
        
        return {
            "current_stage": "evaluated",
            "retry_count": state.get("retry_count", 0) + (1 if result.verdict.value == "revise" else 0),
            "tokens_used": state["tokens_used"] + self.evaluator.total_tokens,
            "cost_usd": state["cost_usd"] + self.evaluator.total_cost,
        }
    
    async def _compile_node(self, state: LearningPlanState) -> dict:
        """Compile the final learning plan document."""
        final_plan = {
            "user_id": state["user_id"],
            "target_role": state["target_role"],
            "created_at": datetime.now().isoformat(),
            "summary": {
                "total_skills": len(state["prioritized_skills"]),
                "estimated_weeks": state.get("deadline_weeks") or 12,
                "hours_per_week": state["available_hours_per_week"],
                "total_hours": state["available_hours_per_week"] * (state.get("deadline_weeks") or 12),
            },
            "skills": state["prioritized_skills"],
            "resources": state["resource_recommendations"],
            "milestones": state["milestone_plan"],
            "projects": state["project_ideas"],
            "next_actions": [
                f"Start with {state['prioritized_skills'][0]['skill']}",
                "Set up your learning environment",
                "Schedule dedicated learning time",
                "Join relevant communities",
            ],
        }
        
        return {
            "final_plan": final_plan,
            "status": "completed",
        }
    
    def _route_after_evaluation(self, state: LearningPlanState) -> str:
        if state.get("retry_count", 0) >= 2:
            return "compile"
        return "compile"
    
    async def run(
        self,
        user_id: str,
        target_role: str,
        current_skills: list[str],
        skill_gaps: list[dict],
        available_hours_per_week: int = 10,
        deadline_weeks: int | None = None,
        learning_style: str = "mixed",
    ) -> dict:
        """Execute the learning plan workflow."""
        initial_state = {
            "user_id": user_id,
            "target_role": target_role,
            "current_skills": current_skills,
            "skill_gaps": skill_gaps,
            "available_hours_per_week": available_hours_per_week,
            "deadline_weeks": deadline_weeks,
            "learning_style": learning_style,
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
        
        result = await self.graph.ainvoke(initial_state)
        return result
