import sys
import os
import io
from pypdf import PdfWriter, PdfReader

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.logic import process_pdf_text, extract_tickets_from_text

def create_dummy_pdf():
    buffer = io.BytesIO()
    p = PdfWriter()
    page = p.add_blank_page(width=200, height=200)
    # pypdf can't write text easily without add_annotation or similar which is complex, 
    # but we can try to assume the user has a PDF. 
    # Actually, we can use fpdf if installed, but it's not in requirements.
    # Alternatively, we can skip creating a PDF and just mock the file object with something that pypdf CAN read if we had a sample.
    # Since we don't have a sample, let's try to make a very basic PDF structure or just fail the test if we can't make one.
    
    # Wait, pypdf is for reading/manipulating. creating from scratch with text is hard.
    # I'll rely on the fact that I can't easily make a PDF here.
    # I will just test the `extract_tickets_from_text` logic with a string, 
    # and separately assume `process_pdf_text` works because it's standard pypdf code.
    return None

def test_groq_extraction():
    print("Testing Groq Extraction...")
    sample_text = """
    POLICY FOR RETURNS
    1. If the item is damaged, you can return it within 30 days.
    2. Refunds are processed to the original payment method.
    
    FAQ
    Q: How do I reset my password?
    A: Go to settings and click reset.
    """
    
    tickets = extract_tickets_from_text(sample_text)
    print(f"Extracted {len(tickets)} tickets.")
    for t in tickets:
        print(f"Issue: {t['issue']}")
        print(f"Resolution: {t['resolution']}")
        print("-" * 20)

if __name__ == "__main__":
    test_groq_extraction()
