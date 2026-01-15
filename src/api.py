from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.logic import find_similar_tickets, generate_answer
from src.pinecone_service import PineconeService

app = FastAPI(title="Knowledge Base API", description="API for Bolna Voice AI to access the knowledge base.")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Knowledge Base API is running"}

@app.get("/search")
def search_knowledge_base(query: str):
    """
    Searches the knowledge base for similar tickets and generates an answer.
    """
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    print(f"Received query: {query}")
    
    # 1. Initialize Pinecone (if not already)
    pc = PineconeService()
    if not pc.index:
        raise HTTPException(status_code=500, detail="Pinecone index not available")

    # 2. Find similar tickets
    # Note: logic.find_similar_tickets handles the embedding generation internally now via PineconeService
    # But wait, find_similar_tickets in logic.py still has the signature (query, tickets, embeddings)
    # let's check logic.py again. 
    # Yes, it takes tickets/embeddings but ignores them in the Pinecone branch. We can pass empty lists.
    
    try:
        similar_results = find_similar_tickets(query, [], None)
    except Exception as e:
         print(f"Error during search: {e}")
         raise HTTPException(status_code=500, detail=str(e))

    if not similar_results:
        return {
            "answer": "I couldn't find any relevant information in the knowledge base.",
            "similar_tickets": []
        }

    # 3. Generate Answer
    try:
        answer = generate_answer(query, similar_results)
    except Exception as e:
        print(f"Error generating answer: {e}")
        answer = "I found some similar tickets but encountered an error generating a summary."

    return {
        "answer": answer,
        "similar_tickets": [res['ticket']['issue'] for res in similar_results]
    }
