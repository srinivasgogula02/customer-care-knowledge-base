import os
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from src.pinecone_service import PineconeService
import pypdf
import json

# Load environment variables
load_dotenv()

def get_secret(key: str) -> str:
    """Get secret from st.secrets (Streamlit Cloud) or os.getenv (local)."""
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except:
        pass
    return os.getenv(key)

# Initialize Groq client
GROQ_API_KEY = get_secret("groq_api_key")
if not GROQ_API_KEY:
    # Try alternate casing just in case
    GROQ_API_KEY = get_secret("GROQ_API_KEY")

client = None
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
else:
    print("Warning: groq_api_key not found in environment or secrets.")

# Global model variable to load only once
_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def generate_embeddings(texts: list[str]):
    model = get_model()
    return model.encode(texts)

def find_similar_tickets(query: str, tickets: list[dict], embeddings: np.ndarray, top_k: int = 3):
    """
    Finds the most similar tickets to the query using Pinecone.
    Note: 'tickets' and 'embeddings' args are kept for signature compatibility but used for fallback or init if needed.
    """
    model = get_model()
    query_embedding = model.encode([query])[0]
    
    # Use Pinecone
    pc_service = PineconeService()
    if pc_service.index:
        return pc_service.search(query_embedding, top_k=top_k)
    else:
        print("Pinecone not available, returning empty results or falling back (not implemented).")
        return []

def generate_answer(query: str, similar_tickets: list[dict]):
    """
    Generates an answer using the Groq API based on similar tickets.
    """
    if not client:
        return "Error: Groq API key not configured."

    context_str = ""
    for item in similar_tickets:
        ticket = item['ticket']
        context_str += f"Issue: {ticket['issue']}\nResolution: {ticket['resolution']}\n---\n"

    system_prompt = (
        "You are an HR knowledge assistant. Provide answers in this exact format:\n\n"
        "**Short Answer:** [One concise sentence answering the question directly]\n\n"
        "**Details:** [Additional relevant information, policy specifics, or steps if applicable]\n\n"
        "Do NOT write emails or greetings. Be direct and factual. "
        "Use the provided context to give accurate answers."
    )

    user_prompt = (
        f"Context from Knowledge Base:\n{context_str}\n\n"
        f"Question: {query}"
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error communicating with Groq API: {str(e)}"

def polish_ticket_content(issue: str, resolution: str):
    """
    Uses the LLM to rewrite the issue and resolution to be clear and professional.
    Returns a tuple (polished_issue, polished_resolution).
    """
    if not client:
        return issue, resolution

    prompt = (
        "You are a helpful assistant. Rewrite the following customer support ticket data to be clear, concise, and professional.\n"
        "Do not change the meaning. Return the result in a valid JSON format with keys 'issue' and 'resolution'.\n\n"
        f"Original Issue: {issue}\n"
        f"Original Resolution: {resolution}\n"
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        content = chat_completion.choices[0].message.content
        import json
        data = json.loads(content)
        return data.get("issue", issue), data.get("resolution", resolution)
    except Exception as e:
        print(f"Error polishing ticket: {e}")
        return issue, resolution

def process_pdf_text(file_obj):
    """
    Extracts text from a PDF file object.
    Returns a list of page contents (one entry per page).
    """
    try:
        pdf = pypdf.PdfReader(file_obj)
        pages = []
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                pages.append({
                    "page_num": i + 1,
                    "content": page_text.strip()
                })
        print(f"Extracted {len(pages)} pages from PDF.")
        return pages
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return []


def extract_tickets_from_text(pages: list):
    """
    Converts PDF pages to ticket format.
    Each page becomes one searchable entry.
    """
    if not pages:
        return []
    
    tickets = []
    for page in pages:
        tickets.append({
            "issue": page["content"],
            "resolution": f"Page {page['page_num']} content.",
            "category": "Document Page"
        })
    
    print(f"Created {len(tickets)} entries (one per page).")
    return tickets


