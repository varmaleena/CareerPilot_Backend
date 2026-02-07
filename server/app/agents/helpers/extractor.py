from app.agents.core.base_agent import BaseAgent
from app.models.agents import AgentRole
from pydantic import BaseModel
import json


class ExtractedResume(BaseModel):
    name: str | None
    email: str | None
    phone: str | None
    summary: str | None
    skills: list[str]
    experience: list[dict]
    education: list[dict]
    certifications: list[str]
    projects: list[dict]


class ExtractorAgent(BaseAgent[ExtractedResume]):
    """Helper agent for data extraction."""
    
    role = AgentRole.EXTRACTOR
    model_tier = "lite"
    max_retries = 3
    timeout_seconds = 10
    
    def get_system_prompt(self) -> str:
        return """You are an Extractor agent. Your job is to parse documents and extract structured data.

Rules:
- Extract only what's present, don't infer or make up data
- Use null for missing fields
- Keep extracted text concise
- Normalize dates to YYYY-MM format

Output valid JSON only."""
    
    def build_prompt(self, document_text: str, extraction_schema: str = "resume") -> str:
        if extraction_schema == "resume":
            schema = """{
    "name": "string or null",
    "email": "string or null",
    "phone": "string or null",
    "summary": "brief professional summary or null",
    "skills": ["skill1", "skill2"],
    "experience": [{"company": "", "title": "", "start": "YYYY-MM", "end": "YYYY-MM or Present", "highlights": []}],
    "education": [{"institution": "", "degree": "", "field": "", "year": ""}],
    "certifications": ["cert1"],
    "projects": [{"name": "", "description": "", "technologies": []}]
}"""
        else:
            schema = extraction_schema
        
        return f"""## Document
{document_text[:8000]}

## Extraction Schema
{schema}

Extract data from the document and respond with valid JSON matching the schema."""
    
    def parse_response(self, response: str) -> ExtractedResume:
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            
            data = json.loads(response.strip())
            return ExtractedResume(**data)
        except Exception as e:
            raise ValueError(f"Failed to parse extraction: {e}")
