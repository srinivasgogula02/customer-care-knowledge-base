import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import load_tickets
from src.logic import generate_embeddings, find_similar_tickets, generate_answer

def test_backend():
    print("Testing Backend Logic...")
    
    # 1. Load Data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tickets.json')
    tickets = load_tickets(data_path)
    print(f"Loaded {len(tickets)} tickets.")
    
    if not tickets:
        print("FAIL: No tickets loaded.")
        return

    # 2. Embeddings
    print("Generating embeddings...")
    corpus = [t['issue'] for t in tickets]
    embeddings = generate_embeddings(corpus)
    print(f"Embeddings shape: {embeddings.shape}")

    # 3. Search
    test_query = "My internet is very slow"
    print(f"Searching for: '{test_query}'")
    results = find_similar_tickets(test_query, tickets, embeddings)
    
    if not results:
        print("FAIL: No results found.")
        return
        
    print(f"Found {len(results)} similar tickets.")
    top_match = results[0]['ticket']['issue']
    print(f"Top match: {top_match}")
    
    # 4. LLM Generation
    print("Generating answer with Groq...")
    answer = generate_answer(test_query, results)
    print("Generated Answer:")
    print(answer)
    print("SUCCESS: Backend verification complete.")

if __name__ == "__main__":
    test_backend()
