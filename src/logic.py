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

# Initialize Groq client
GROQ_API_KEY = os.getenv("groq_api_key")
if not GROQ_API_KEY:
    # Try alternate casing just in case
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = None
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
else:
    print("Warning: groq_api_key not found in environment variables.")

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
        "You are a helpful customer support assistant. "
        "Use the following similar resolved tickets to suggest a resolution for the new issue. "
        "If the similar tickets are not relevant, use your general knowledge but mention that you are doing so. "
        "Be polite and professional."
    )

    user_prompt = (
        f"Context (Similar Past Tickets):\n{context_str}\n\n"
        f"New Customer Issue:\n{query}\n\n"
        "Suggested Response:"
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
    """
    try:
        pdf = pypdf.PdfReader(file_obj)
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # Handle None return from extract_text()
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def extract_tickets_from_text(text: str):
    """
    Uses Groq LLM to extract Question-Answer pairs from document text.
    Returns a list of dicts with 'issue' (question) and 'resolution' (answer).
    """
    # Validate input
    if not text or not text.strip():
        print("Warning: Empty or whitespace-only text passed to extract_tickets_from_text.")
        return []
    
    if not client:
        print("Error: Groq client not initialized.")
        return []
        
    text = text.strip()
    tickets = []
    
    # Chunk document for processing (4000 chars for more context)
    chunk_size = 4000
    overlap = 500
    text_len = len(text)
    start = 0
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        start += (chunk_size - overlap)
        
        if not chunk:
            continue
            
        prompt = (
            "You are an expert at extracting FAQ and Q&A pairs from documents.\n\n"
            "Analyze the following text and extract ONLY explicit Question-Answer pairs.\n"
            "Look for patterns like 'Q:', 'Q.', 'Question:', or any clear question followed by its answer.\n"
            "Also extract policy statements and convert them to Q&A format.\n\n"
            "Rules:\n"
            "- Each 'issue' should be a clear, concise QUESTION (not raw text)\n"
            "- Each 'resolution' should be a clear, concise ANSWER\n"
            "- Do NOT include raw paragraph text\n"
            "- If no Q&A pairs exist in this chunk, return empty tickets array\n\n"
            "Return JSON: {\"tickets\": [{\"issue\": \"question\", \"resolution\": \"answer\"}, ...]}\n\n"
            f"Document Text:\n{chunk}"
        )

        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful API that outputs strictly JSON."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            content = chat_completion.choices[0].message.content
            data = json.loads(content)
            chunk_tickets = data.get("tickets", [])
            
            # Filter out any entries that look like raw chunks
            for t in chunk_tickets:
                issue = t.get('issue', '')
                resolution = t.get('resolution', '')
                # Only add if both fields are reasonable length and resolution isn't placeholder
                if issue and resolution and len(issue) < 500 and resolution != "Content from source document.":
                    tickets.append(t)
                    
            print(f"Extracted {len(chunk_tickets)} Q&A pairs from chunk.")
        except Exception as e:
            print(f"Error extracting from chunk: {e}")
            continue

    print(f"Total Q&A pairs extracted: {len(tickets)}")
    return tickets
