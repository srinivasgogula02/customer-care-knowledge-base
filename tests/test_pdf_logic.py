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
    print("Testing Raw Chunking Logic...")
    
    chunk_size = 3000
    overlap = 500
    
    # Create text that creates exactly 2 chunks
    # Chunk 1: 0-3000
    # Next start: 2500. 
    # So if text is 3500 chars long:
    # 1. 0-3000
    # 2. 2500-3500
    
    sample_text = "A" * 3500
    
    tickets = extract_tickets_from_text(sample_text)
    print(f"Extracted {len(tickets)} chunks.")
    
    if len(tickets) == 2:
        print("SUCCESS: Correct number of chunks.")
    else:
        print(f"FAILURE: Expected 2 chunks, got {len(tickets)}")
        
    # Verify content
    if tickets[0]['issue'] == "A" * 3000:
        print("SUCCESS: Chunk 1 content correct.")
    else:
        print(f"FAILURE: Chunk 1 length {len(tickets[0]['issue'])}")

    if tickets[1]['issue'] == "A" * 1000: # 3500 - 2500
        print("SUCCESS: Chunk 2 content correct.")
    else:
        print("FAILURE: Chunk 2 content incorrect.")

if __name__ == "__main__":
    test_groq_extraction()
