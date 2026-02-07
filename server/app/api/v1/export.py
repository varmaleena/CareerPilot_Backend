from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from io import BytesIO

from app.api.deps import CurrentUser
from app.services.export.pdf_generator import PDFExporter, LaTeXGenerator

router = APIRouter()


class ExportRequest(BaseModel):
    resume_data: dict = Field(..., description="Structured resume data from analysis")
    format: str = Field("pdf", pattern="^(pdf|latex)$")


@router.post("/resume")
async def export_resume(
    request: ExportRequest,
    user: CurrentUser,
):
    """Export resume to PDF or LaTeX format."""
    
    if request.format == "latex":
        generator = LaTeXGenerator()
        latex_content = generator.generate_latex(request.resume_data)
        return Response(
            content=latex_content,
            media_type="application/x-tex",
            headers={"Content-Disposition": "attachment; filename=resume.tex"}
        )
    
    # PDF export
    exporter = PDFExporter()
    pdf_bytes = await exporter.export_resume_to_pdf(request.resume_data)
    
    # If pdflatex not available, returns LaTeX bytes
    if isinstance(pdf_bytes, bytes) and pdf_bytes.startswith(b"\\documentclass"):
        return Response(
            content=pdf_bytes,
            media_type="application/x-tex",
            headers={
                "Content-Disposition": "attachment; filename=resume.tex",
                "X-Fallback": "true",
            }
        )
    
    return StreamingResponse(
        BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=resume.pdf"}
    )


@router.post("/learning-plan")
async def export_learning_plan(
    plan_data: dict,
    user: CurrentUser,
):
    """Export learning plan to PDF."""
    # TODO: Implement learning plan PDF template
    return {"status": "not_implemented", "message": "Learning plan export coming soon"}
