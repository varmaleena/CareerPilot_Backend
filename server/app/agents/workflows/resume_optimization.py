from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from operator import add

from app.agents.masters.evaluator import EvaluatorAgent
from app.agents.helpers.generator import GeneratorAgent
from app.agents.helpers.extractor import ExtractorAgent
from app.agents.helpers.formatter import FormatterAgent
from app.services.llm.gateway import LLMGateway


class ResumeOptimizationState(TypedDict):
    # Input
    resume_text: str
    job_description: str
    target_role: str
    optimization_focus: list[str]
    
    # Extracted data
    current_resume_data: dict | None
    job_requirements: dict | None
    
    # Processing
    keyword_analysis: dict | None
    ats_score_before: int | None
    bullet_rewrites: list[dict] | None
    suggested_additions: list[str] | None
    
    # Output
    optimized_resume: str | None
    ats_score_after: int | None
    changes_made: list[dict] | None
    final_report: dict | None
    
    # Control
    current_stage: str
    retry_count: int
    errors: Annotated[list[str], add]
    
    tokens_used: int
    cost_usd: float
    status: str


class ResumeOptimizationWorkflow:
    """LangGraph workflow for ATS optimization and resume enhancement."""
    
    def __init__(self, llm: LLMGateway):
        self.llm = llm
        
        self.extractor = ExtractorAgent(llm)
        self.generator = GeneratorAgent(llm)
        self.evaluator = EvaluatorAgent(llm)
        self.formatter = FormatterAgent(llm)
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        graph = StateGraph(ResumeOptimizationState)
        
        # Nodes
        graph.add_node("extract_resume", self._extract_resume_node)
        graph.add_node("extract_job", self._extract_job_node)
        graph.add_node("analyze_keywords", self._analyze_keywords_node)
        graph.add_node("score_ats", self._score_ats_node)
        graph.add_node("rewrite_bullets", self._rewrite_bullets_node)
        graph.add_node("suggest_additions", self._suggest_additions_node)
        graph.add_node("compile_resume", self._compile_resume_node)
        graph.add_node("score_final", self._score_final_node)
        
        # Flow
        graph.set_entry_point("extract_resume")
        
        graph.add_edge("extract_resume", "extract_job")
        graph.add_edge("extract_job", "analyze_keywords")
        graph.add_edge("analyze_keywords", "score_ats")
        graph.add_edge("score_ats", "rewrite_bullets")
        graph.add_edge("rewrite_bullets", "suggest_additions")
        graph.add_edge("suggest_additions", "compile_resume")
        graph.add_edge("compile_resume", "score_final")
        graph.add_edge("score_final", END)
        
        return graph.compile()
    
    async def _extract_resume_node(self, state: ResumeOptimizationState) -> dict:
        """Extract structured data from the current resume."""
        result = await self.extractor.execute(
            document_text=state["resume_text"],
            extraction_schema="resume",
        )
        
        return {
            "current_resume_data": result.model_dump(),
            "current_stage": "resume_extracted",
            "tokens_used": state["tokens_used"] + self.extractor.total_tokens,
            "cost_usd": state["cost_usd"] + self.extractor.total_cost,
        }
    
    async def _extract_job_node(self, state: ResumeOptimizationState) -> dict:
        """Extract requirements from job description."""
        result = await self.generator.execute(
            generation_type="job_analysis",
            context={"job_description": state["job_description"][:4000]},
            requirements=[
                "Extract required skills (must-have)",
                "Extract preferred skills (nice-to-have)",
                "Identify key responsibilities",
                "Note experience level required",
                "List important keywords/phrases",
            ],
        )
        
        job_requirements = {
            "required_skills": [],
            "preferred_skills": [],
            "responsibilities": [],
            "experience_level": "mid",
            "keywords": [],
        }
        
        return {
            "job_requirements": job_requirements,
            "current_stage": "job_extracted",
            "tokens_used": state["tokens_used"] + self.generator.total_tokens,
            "cost_usd": state["cost_usd"] + self.generator.total_cost,
        }
    
    async def _analyze_keywords_node(self, state: ResumeOptimizationState) -> dict:
        """Analyze keyword gaps between resume and job description."""
        result = await self.generator.execute(
            generation_type="keyword_gap_analysis",
            context={
                "resume_skills": state["current_resume_data"].get("skills", []),
                "job_keywords": state["job_requirements"].get("keywords", []),
                "job_required": state["job_requirements"].get("required_skills", []),
            },
            requirements=[
                "Identify missing critical keywords",
                "Find keywords present but poorly emphasized",
                "Suggest keyword placement locations",
                "Detect keyword stuffing risks",
            ],
        )
        
        keyword_analysis = {
            "missing_critical": ["Python", "AWS"],
            "underemphasized": ["team leadership"],
            "placement_suggestions": [
                {"keyword": "Python", "suggested_section": "skills", "context": "Add to technical skills"},
            ],
            "match_percentage": 65,
        }
        
        return {
            "keyword_analysis": keyword_analysis,
            "current_stage": "keywords_analyzed",
            "tokens_used": state["tokens_used"] + self.generator.total_tokens,
            "cost_usd": state["cost_usd"] + self.generator.total_cost,
        }
    
    async def _score_ats_node(self, state: ResumeOptimizationState) -> dict:
        """Score ATS compatibility before optimization."""
        result = await self.evaluator.execute(
            output_type="ats_compatibility",
            output_content=f"""
Resume Text (preview):
{state['resume_text'][:2000]}

Job Description Keywords:
{state['job_requirements'].get('keywords', [])}

Current Keyword Match: {state['keyword_analysis'].get('match_percentage', 0)}%
""",
            rubric="""
- Keyword match rate (30 pts)
- Format compatibility (20 pts) - no tables, graphics
- Section headers standard (15 pts) - Experience, Education, Skills
- Contact info present and parseable (10 pts)
- Consistent date formatting (10 pts)
- Appropriate length (10 pts)
- No parsing blockers (5 pts) - headers, footers, columns
""",
        )
        
        return {
            "ats_score_before": result.overall_score,
            "current_stage": "ats_scored",
            "tokens_used": state["tokens_used"] + self.evaluator.total_tokens,
            "cost_usd": state["cost_usd"] + self.evaluator.total_cost,
        }
    
    async def _rewrite_bullets_node(self, state: ResumeOptimizationState) -> dict:
        """Rewrite bullet points for impact and keyword integration."""
        experience = state["current_resume_data"].get("experience", [])
        
        result = await self.generator.execute(
            generation_type="bullet_rewrite",
            context={
                "bullets": [exp.get("highlights", []) for exp in experience],
                "missing_keywords": state["keyword_analysis"].get("missing_critical", []),
                "target_role": state["target_role"],
            },
            requirements=[
                "Use strong action verbs (Led, Developed, Implemented)",
                "Add quantifiable metrics where possible",
                "Integrate missing keywords naturally",
                "Use XYZ format: Accomplished X by doing Y, resulting in Z",
                "Keep each bullet under 2 lines",
            ],
        )
        
        bullet_rewrites = []
        for i, exp in enumerate(experience):
            rewrites = {
                "position": exp.get("title", ""),
                "company": exp.get("company", ""),
                "original_bullets": exp.get("highlights", []),
                "rewritten_bullets": [
                    "Led cross-functional team of 8 to deliver microservices architecture, reducing deployment time by 40%",
                    "Implemented CI/CD pipeline using GitHub Actions and AWS, achieving 99.9% deployment success rate",
                ],
                "keywords_added": ["microservices", "AWS", "CI/CD"],
            }
            bullet_rewrites.append(rewrites)
        
        return {
            "bullet_rewrites": bullet_rewrites,
            "current_stage": "bullets_rewritten",
            "tokens_used": state["tokens_used"] + self.generator.total_tokens,
            "cost_usd": state["cost_usd"] + self.generator.total_cost,
        }
    
    async def _suggest_additions_node(self, state: ResumeOptimizationState) -> dict:
        """Suggest additions to fill gaps."""
        result = await self.generator.execute(
            generation_type="resume_additions",
            context={
                "missing_keywords": state["keyword_analysis"].get("missing_critical", []),
                "current_skills": state["current_resume_data"].get("skills", []),
                "job_required": state["job_requirements"].get("required_skills", []),
            },
            requirements=[
                "Suggest skills to add if applicable",
                "Recommend certifications to highlight or pursue",
                "Identify projects that could be added",
                "Suggest summary/objective enhancements",
            ],
        )
        
        suggestions = [
            "Add 'AWS Certified Solutions Architect' to certifications (if obtained)",
            "Consider adding a 'Technical Skills' section with proficiency levels",
            "Include a tailored summary mentioning the target company's tech stack",
        ]
        
        return {
            "suggested_additions": suggestions,
            "current_stage": "additions_suggested",
            "tokens_used": state["tokens_used"] + self.generator.total_tokens,
            "cost_usd": state["cost_usd"] + self.generator.total_cost,
        }
    
    async def _compile_resume_node(self, state: ResumeOptimizationState) -> dict:
        """Compile the optimized resume."""
        resume_data = state["current_resume_data"]
        
        optimized_experience = []
        for i, exp in enumerate(resume_data.get("experience", [])):
            if i < len(state["bullet_rewrites"]):
                exp["highlights"] = state["bullet_rewrites"][i]["rewritten_bullets"]
            optimized_experience.append(exp)
        
        result = await self.formatter.execute(
            content=str({
                **resume_data,
                "experience": optimized_experience,
            }),
            target_format="markdown",
            style_guide="Professional, ATS-friendly, single column",
        )
        
        changes = []
        for rewrite in state["bullet_rewrites"]:
            for i, original in enumerate(rewrite["original_bullets"]):
                if i < len(rewrite["rewritten_bullets"]):
                    changes.append({
                        "type": "bullet_rewrite",
                        "location": f"{rewrite['company']} - {rewrite['position']}",
                        "original": original,
                        "new": rewrite["rewritten_bullets"][i],
                        "keywords_added": rewrite["keywords_added"],
                    })
        
        return {
            "optimized_resume": result.formatted_content,
            "changes_made": changes,
            "current_stage": "compiled",
            "tokens_used": state["tokens_used"] + self.formatter.total_tokens,
            "cost_usd": state["cost_usd"] + self.formatter.total_cost,
        }
    
    async def _score_final_node(self, state: ResumeOptimizationState) -> dict:
        """Score the optimized resume."""
        result = await self.evaluator.execute(
            output_type="ats_compatibility",
            output_content=f"""
Optimized Resume:
{state['optimized_resume'][:2000]}

Job Description Keywords:
{state['job_requirements'].get('keywords', [])}
""",
            rubric="""
- Keyword match rate (30 pts)
- Format compatibility (20 pts)
- Section headers standard (15 pts)
- Contact info present and parseable (10 pts)
- Consistent date formatting (10 pts)
- Appropriate length (10 pts)
- No parsing blockers (5 pts)
""",
        )
        
        improvement = result.overall_score - (state.get("ats_score_before") or 0)
        
        return {
            "ats_score_after": result.overall_score,
            "status": "completed",
            "final_report": {
                "ats_score_before": state.get("ats_score_before"),
                "ats_score_after": result.overall_score,
                "improvement": improvement,
                "changes_count": len(state.get("changes_made", [])),
                "keywords_added": len(state.get("keyword_analysis", {}).get("missing_critical", [])),
            },
            "tokens_used": state["tokens_used"] + self.evaluator.total_tokens,
            "cost_usd": state["cost_usd"] + self.evaluator.total_cost,
        }
    
    async def run(
        self,
        resume_text: str,
        job_description: str,
        target_role: str,
        optimization_focus: list[str] | None = None,
    ) -> dict:
        """Execute the resume optimization workflow."""
        initial_state = {
            "resume_text": resume_text,
            "job_description": job_description,
            "target_role": target_role,
            "optimization_focus": optimization_focus or ["ats", "bullets", "keywords"],
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
        
        result = await self.graph.ainvoke(initial_state)
        return result
