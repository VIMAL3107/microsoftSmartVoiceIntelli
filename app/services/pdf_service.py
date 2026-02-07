
import fitz  # PyMuPDF
import os
from fpdf import FPDF
import tempfile
from pathlib import Path

def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

def generate_call_pdf_report(session_id, analyzed_text, performance_summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"Call Report - Session ID: {session_id}", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Translate:", ln=True)
    pdf.set_font("Arial", "", 12)
    # Ensure text is not None
    text_content = analyzed_text if analyzed_text else "No translation available."
    pdf.multi_cell(0, 8, text_content)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Agent Performance Summary:", ln=True)
    pdf.set_font("Arial", "", 12)
    summary_content = performance_summary if performance_summary else "No feedback available."
    pdf.multi_cell(0, 8, summary_content)

    # Save to a temporary directory
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, f"{session_id}_report.pdf")
    pdf.output(pdf_path)
    
    return pdf_path
