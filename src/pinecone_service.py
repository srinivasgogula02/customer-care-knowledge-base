import os
import time
import hashlib
from typing import List, Dict, Union
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

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

class PineconeService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PineconeService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.api_key = get_secret("pinecone_api")
        if not self.api_key:
            print("Error: pinecone_api key not found in environment or secrets.")
            self.index = None
            return

        try:
            self.pc = Pinecone(api_key=self.api_key)
            self.index_name = "tickets-index"
            self.dimension = 384 # all-MiniLM-L6-v2 dimension
            self.metric = "cosine"
            self._ensure_index_exists()
            self.index = self.pc.Index(self.index_name)
            self._initialized = True
            print("Pinecone initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize Pinecone: {e}")
            self.index = None

    def _ensure_index_exists(self):
        try:
            existing_indexes = [i.name for i in self.pc.list_indexes()]
            if self.index_name not in existing_indexes:
                print(f"Creating index {self.index_name}...")
                try:
                    # Attempt to create a serverless index (available on free tier normally)
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        metric=self.metric,
                        spec=ServerlessSpec(
                            cloud="aws",
                            region="us-east-1"
                        )
                    )
                except Exception as e:
                    print(f"Note: Serverless index creation failed ({e}). Please ensure your Pinecone plan supports this configuration.")
                    raise e
                    
                # Wait for index to be ready
                while not self.pc.describe_index(self.index_name).status['ready']:
                    time.sleep(1)
                print(f"Index {self.index_name} created.")
            else:
                print(f"Index {self.index_name} already exists.")
        except Exception as e:
            print(f"Error checking/creating index: {e}")
            # Do not raise here to allow app to continue if index check fails (maybe network issue), 
            # though functionality will be impaired.
            pass

    def generate_id(self, ticket: Dict) -> str:
        """Generate a deterministic ID based on ticket content."""
        content = f"{ticket.get('issue', '')}{ticket.get('resolution', '')}"
        return hashlib.md5(content.encode()).hexdigest()

    def upsert_tickets(self, tickets: List[Dict], embeddings: Union[List[List[float]], np.ndarray]) -> bool:
        if not self.index:
            print("Pinecone index not initialized.")
            return False

        vectors = []
        for i, ticket in enumerate(tickets):
            # Generate ID
            ticket_id = self.generate_id(ticket)
            
            # Prepare metadata (truncate issue to avoid Pinecone 40KB limit)
            issue_text = ticket.get("issue", "")
            if len(issue_text) > 4000:
                issue_text = issue_text[:4000] + "..."
                
            metadata = {
                "category": ticket.get("category", "General"),
                "issue": issue_text,
                "resolution": ticket.get("resolution", "")[:2000],  # Also truncate resolution
                "source": ticket.get("source", "Manual")
            }
            
            # Handle numpy array if passed
            values = embeddings[i]
            if isinstance(values, np.ndarray):
                values = values.tolist()
            
            vectors.append({
                "id": ticket_id,
                "values": values,
                "metadata": metadata
            })

        # Batch upsert
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            try:
                self.index.upsert(vectors=batch)
                print(f"Upserted batch {i//batch_size + 1}/{len(vectors)//batch_size + 1}")
            except Exception as e:
                print(f"Error upserting batch: {e}")
                return False
        return True

    def search(self, query_embedding: Union[List[float], np.ndarray], top_k: int = 3) -> List[Dict]:
        if not self.index:
            print("Pinecone index not initialized, cannot search.")
            return []

        try:
            # Handle numpy array
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
                
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            formatted_results = []
            for match in results['matches']:
                # Ensure metadata is present
                meta = match.get('metadata', {})
                ticket = {
                    "issue": meta.get('issue', 'N/A'),
                    "resolution": meta.get('resolution', 'N/A'),
                    "category": meta.get('category', 'General'),
                    "source": meta.get('source', 'Manual')
                }
                formatted_results.append({
                    "ticket": ticket,
                    "score": match['score']
                })
            return formatted_results
        except Exception as e:
            print(f"Error searching Pinecone: {e}")
            return []

    def delete_all_vectors(self) -> bool:
        """Delete all vectors from the index."""
        if not self.index:
            print("Pinecone index not initialized, cannot delete.")
            return False

        try:
            self.index.delete(delete_all=True)
            print(f"All vectors deleted from index {self.index_name}.")
            return True
        except Exception as e:
            print(f"Error deleting vectors: {e}")
            return False
