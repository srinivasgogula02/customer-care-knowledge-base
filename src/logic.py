import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv

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
    Finds the most similar tickets to the query.
    """
    model = get_model()
    query_embedding = model.encode([query])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "ticket": tickets[idx],
            "score": similarities[idx]
        })
    
    return results

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
