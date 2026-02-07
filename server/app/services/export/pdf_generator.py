import subprocess
import tempfile
import os
from pathlib import Path
from app.services.llm.gateway import LLMGateway


class LaTeXGenerator:
    """Generate LaTeX documents from structured data."""
    
    RESUME_TEMPLATE = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\usepackage{enumitem}
\usepackage{hyperref}
\usepackage{titlesec}
\usepackage{xcolor}

\geometry{left=0.75in, right=0.75in, top=0.5in, bottom=0.5in}
\pagestyle{empty}

\definecolor{headercolor}{RGB}{0, 51, 102}
\titleformat{\section}{\large\bfseries\color{headercolor}}{}{0em}{}[\titlerule]

\begin{document}

% Header
\begin{center}
    {\LARGE\bfseries <<NAME>>} \\[2pt]
    <<EMAIL>> $|$ <<PHONE>> $|$ <<LOCATION>>
\end{center}

% Summary
<<SUMMARY_SECTION>>

% Experience
\section{Experience}
<<EXPERIENCE_SECTION>>

% Education
\section{Education}
<<EDUCATION_SECTION>>

% Skills
\section{Skills}
<<SKILLS_SECTION>>

% Projects
<<PROJECTS_SECTION>>

% Certifications
<<CERTIFICATIONS_SECTION>>

\end{document}
"""
    
    def __init__(self, llm: LLMGateway | None = None):
        self.llm = llm
    
    def generate_latex(self, resume_data: dict) -> str:
        """Generate LaTeX code from structured resume data."""
        latex = self.RESUME_TEMPLATE
        
        # Header
        latex = latex.replace("<<NAME>>", self._escape(resume_data.get("name", "Your Name")))
        latex = latex.replace("<<EMAIL>>", self._escape(resume_data.get("email", "")))
        latex = latex.replace("<<PHONE>>", self._escape(resume_data.get("phone", "")))
        latex = latex.replace("<<LOCATION>>", self._escape(resume_data.get("location", "")))
        
        # Summary
        summary = resume_data.get("summary", "")
        if summary:
            summary_section = f"\\section{{Summary}}\n{self._escape(summary)}\n"
        else:
            summary_section = ""
        latex = latex.replace("<<SUMMARY_SECTION>>", summary_section)
        
        # Experience
        experience_items = []
        for exp in resume_data.get("experience", []):
            exp_latex = self._format_experience(exp)
            experience_items.append(exp_latex)
        latex = latex.replace("<<EXPERIENCE_SECTION>>", "\n\n".join(experience_items))
        
        # Education
        education_items = []
        for edu in resume_data.get("education", []):
            edu_latex = self._format_education(edu)
            education_items.append(edu_latex)
        latex = latex.replace("<<EDUCATION_SECTION>>", "\n\n".join(education_items))
        
        # Skills
        skills = resume_data.get("skills", [])
        if skills:
            skills_latex = ", ".join([self._escape(s) for s in skills])
        else:
            skills_latex = "No skills listed"
        latex = latex.replace("<<SKILLS_SECTION>>", skills_latex)
        
        # Projects
        projects = resume_data.get("projects", [])
        if projects:
            projects_section = "\\section{Projects}\n"
            for proj in projects:
                projects_section += self._format_project(proj) + "\n\n"
        else:
            projects_section = ""
        latex = latex.replace("<<PROJECTS_SECTION>>", projects_section)
        
        # Certifications
        certs = resume_data.get("certifications", [])
        if certs:
            certs_section = "\\section{Certifications}\n"
            certs_section += ", ".join([self._escape(c) for c in certs])
        else:
            certs_section = ""
        latex = latex.replace("<<CERTIFICATIONS_SECTION>>", certs_section)
        
        return latex
    
    def _format_experience(self, exp: dict) -> str:
        title = self._escape(exp.get("title", ""))
        company = self._escape(exp.get("company", ""))
        start = self._escape(exp.get("start", ""))
        end = self._escape(exp.get("end", "Present"))
        
        latex = f"\\textbf{{{title}}} \\hfill {start} -- {end} \\\\\n"
        latex += f"\\textit{{{company}}}\n"
        
        highlights = exp.get("highlights", [])
        if highlights:
            latex += "\\begin{itemize}[leftmargin=*, nosep]\n"
            for h in highlights:
                latex += f"    \\item {self._escape(h)}\n"
            latex += "\\end{itemize}"
        
        return latex
    
    def _format_education(self, edu: dict) -> str:
        institution = self._escape(edu.get("institution", ""))
        degree = self._escape(edu.get("degree", ""))
        field = self._escape(edu.get("field", ""))
        year = self._escape(edu.get("year", ""))
        
        latex = f"\\textbf{{{degree} in {field}}} \\hfill {year} \\\\\n"
        latex += f"\\textit{{{institution}}}"
        
        return latex
    
    def _format_project(self, proj: dict) -> str:
        name = self._escape(proj.get("name", ""))
        description = self._escape(proj.get("description", ""))
        technologies = proj.get("technologies", [])
        
        latex = f"\\textbf{{{name}}}"
        if technologies:
            latex += f" \\textit{{({', '.join([self._escape(t) for t in technologies])})}}"
        latex += f" \\\\\n{description}"
        
        return latex
    
    def _escape(self, text: str) -> str:
        """Escape special LaTeX characters."""
        if not text:
            return ""
        replacements = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}',
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        return text


class PDFExporter:
    """Convert LaTeX to PDF using pdflatex."""
    
    def __init__(self):
        self.latex_generator = LaTeXGenerator()
    
    async def export_resume_to_pdf(
        self,
        resume_data: dict,
        output_path: str | None = None,
    ) -> bytes | str:
        """Generate PDF from resume data."""
        # Generate LaTeX
        latex_content = self.latex_generator.generate_latex(resume_data)
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = Path(tmpdir) / "resume.tex"
            pdf_path = Path(tmpdir) / "resume.pdf"
            
            # Write LaTeX file
            tex_path.write_text(latex_content)
            
            # Run pdflatex
            try:
                result = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, str(tex_path)],
                    capture_output=True,
                    timeout=30,
                )
                
                if pdf_path.exists():
                    pdf_bytes = pdf_path.read_bytes()
                    
                    # If output path specified, write to file
                    if output_path:
                        Path(output_path).write_bytes(pdf_bytes)
                        return output_path
                    
                    return pdf_bytes
                else:
                    # Return LaTeX if pdflatex not available
                    return latex_content.encode()
                    
            except FileNotFoundError:
                # pdflatex not installed, return LaTeX
                return latex_content.encode()
            except subprocess.TimeoutExpired:
                raise RuntimeError("PDF generation timed out")
    
    def export_latex(self, resume_data: dict) -> str:
        """Export LaTeX source only."""
        return self.latex_generator.generate_latex(resume_data)


# Convenience functions
async def generate_resume_pdf(resume_data: dict) -> bytes:
    """Generate PDF resume from structured data."""
    exporter = PDFExporter()
    return await exporter.export_resume_to_pdf(resume_data)


def generate_resume_latex(resume_data: dict) -> str:
    """Generate LaTeX resume source from structured data."""
    generator = LaTeXGenerator()
    return generator.generate_latex(resume_data)
