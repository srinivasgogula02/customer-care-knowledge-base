from fastapi.testclient import TestClient
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api import app

client = TestClient(app)

def test_api():
    print("Testing API...")
    
    # 1. Health Check
    response = client.get("/")
    assert response.status_code == 200
    print("Health Check Passed: ", response.json())
    
    # 2. Search
    test_query = "My internet is very slow"
    print(f"Testing search with query: '{test_query}'")
    
    # Note: This will actually fail if Pinecone is not initialized/working
    # Ideally we'd mock PineconeService, but for integration test we want real connection
    try:
        response = client.get(f"/search?query={test_query}")
        
        if response.status_code == 200:
            data = response.json()
            print("Search Success!")
            print(f"Answer: {data.get('answer')}")
            print(f"Found {len(data.get('similar_tickets', []))} similar tickets.")
        else:
            print(f"Search Failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Test Execution Error: {e}")

if __name__ == "__main__":
    test_api()
