from pydantic import BaseModel
from app.services.llm.gateway import LLMGateway


class MarketInsight(BaseModel):
    role: str
    demand_score: int  # 0-100
    salary_range: dict  # {"min": 80000, "max": 150000, "median": 115000}
    top_skills: list[str]
    emerging_skills: list[str]
    job_growth_rate: float  # Percentage
    remote_availability: float  # Percentage
    top_locations: list[str]
    industry_trends: list[str]


class MarketService:
    """Analyze job market trends and provide insights."""
    
    def __init__(self, llm: LLMGateway | None = None):
        self.llm = llm
    
    async def get_role_insights(self, role: str, location: str = "") -> MarketInsight:
        """Get market insights for a specific role."""
        # In production, this would aggregate real data
        # For now, return structured mock data
        
        insights = {
            "software engineer": MarketInsight(
                role="Software Engineer",
                demand_score=85,
                salary_range={"min": 90000, "max": 180000, "median": 130000},
                top_skills=["Python", "JavaScript", "AWS", "React", "SQL"],
                emerging_skills=["Rust", "AI/ML", "Kubernetes", "LangChain"],
                job_growth_rate=15.2,
                remote_availability=75.0,
                top_locations=["San Francisco", "New York", "Seattle", "Austin"],
                industry_trends=[
                    "AI/ML integration in all products",
                    "Shift to cloud-native development",
                    "Increased focus on developer experience",
                ],
            ),
            "data scientist": MarketInsight(
                role="Data Scientist",
                demand_score=80,
                salary_range={"min": 100000, "max": 200000, "median": 145000},
                top_skills=["Python", "SQL", "Machine Learning", "Statistics", "TensorFlow"],
                emerging_skills=["LLMs", "MLOps", "Generative AI", "Real-time ML"],
                job_growth_rate=22.0,
                remote_availability=65.0,
                top_locations=["San Francisco", "New York", "Boston", "Seattle"],
                industry_trends=[
                    "GenAI transforming data science workflows",
                    "Increased demand for MLOps skills",
                    "Focus on explainable AI",
                ],
            ),
            "product manager": MarketInsight(
                role="Product Manager",
                demand_score=75,
                salary_range={"min": 110000, "max": 220000, "median": 155000},
                top_skills=["Product Strategy", "Data Analysis", "User Research", "Agile", "SQL"],
                emerging_skills=["AI Product Management", "Growth Hacking", "PLG"],
                job_growth_rate=12.0,
                remote_availability=55.0,
                top_locations=["San Francisco", "New York", "Seattle", "Los Angeles"],
                industry_trends=[
                    "AI-first product development",
                    "Product-led growth strategies",
                    "Emphasis on data-driven decisions",
                ],
            ),
        }
        
        role_lower = role.lower()
        for key, insight in insights.items():
            if key in role_lower or role_lower in key:
                return insight
        
        # Default fallback
        return MarketInsight(
            role=role,
            demand_score=70,
            salary_range={"min": 80000, "max": 150000, "median": 110000},
            top_skills=["Communication", "Problem Solving", "Technical Skills"],
            emerging_skills=["AI Tools", "Automation", "Data Literacy"],
            job_growth_rate=8.0,
            remote_availability=45.0,
            top_locations=["Major Metro Areas"],
            industry_trends=["Digital transformation across industries"],
        )
    
    async def compare_skills_to_market(
        self,
        user_skills: list[str],
        target_role: str,
    ) -> dict:
        """Compare user skills to market requirements."""
        insights = await self.get_role_insights(target_role)
        
        user_skills_lower = {s.lower() for s in user_skills}
        top_skills_lower = {s.lower() for s in insights.top_skills}
        emerging_lower = {s.lower() for s in insights.emerging_skills}
        
        matching = user_skills_lower & top_skills_lower
        missing = top_skills_lower - user_skills_lower
        emerging_gap = emerging_lower - user_skills_lower
        
        # Calculate readiness score
        readiness = len(matching) / max(len(top_skills_lower), 1) * 100
        
        return {
            "role": target_role,
            "readiness_score": int(readiness),
            "matching_skills": list(matching),
            "missing_core_skills": list(missing),
            "emerging_skills_gap": list(emerging_gap),
            "market_insight": insights.model_dump(),
            "recommendations": [
                f"Learn {skill}" for skill in list(missing)[:3]
            ] + [
                f"Explore emerging skill: {skill}" for skill in list(emerging_gap)[:2]
            ],
        }
    
    async def salary_benchmark(
        self,
        role: str,
        experience_years: int,
        location: str = "",
    ) -> dict:
        """Get salary benchmarks for a role."""
        insights = await self.get_role_insights(role)
        base = insights.salary_range
        
        # Adjust for experience
        experience_multiplier = 1 + (experience_years * 0.05)  # 5% per year
        
        return {
            "role": role,
            "experience_years": experience_years,
            "salary_range": {
                "min": int(base["min"] * experience_multiplier),
                "max": int(base["max"] * experience_multiplier),
                "median": int(base["median"] * experience_multiplier),
            },
            "percentiles": {
                "25th": int(base["min"] * experience_multiplier * 1.1),
                "50th": int(base["median"] * experience_multiplier),
                "75th": int(base["max"] * experience_multiplier * 0.9),
                "90th": int(base["max"] * experience_multiplier),
            },
        }


market_service = MarketService()
