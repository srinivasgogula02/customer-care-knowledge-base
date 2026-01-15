import json
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pinecone_service import PineconeService
from src.logic import generate_embeddings

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tickets.json')
SEED_PATH = os.path.join(os.path.dirname(__file__), '..', 'seeds', 'educational_institute_data.json')

def seed_data():
    print("Starting data seeding...")
    
    # 1. Load Seed Data
    if not os.path.exists(SEED_PATH):
        print(f"Error: Seed file not found at {SEED_PATH}")
        return

    with open(SEED_PATH, 'r') as f:
        new_tickets = json.load(f)
    print(f"Loaded {len(new_tickets)} tickets from seed file.")

    # 2. Update JSON File (Overwrite)
    with open(DATA_PATH, 'w') as f:
        json.dump(new_tickets, f, indent=2)
    print(f"Updated {DATA_PATH} with new data.")

    # 3. Update Pinecone
    print("Syncing with Pinecone...")
    try:
        # Initialize Pinecone Service
        pc = PineconeService()
        if not pc.index:
            print("Error: Pinecone index not initialized.")
            return

        # Generate Embeddings
        print("Generating embeddings...")
        corpus = [t['issue'] for t in new_tickets]
        embeddings = generate_embeddings(corpus)
        
        # Upsert to Pinecone
        print("Upserting vectors...")
        success = pc.upsert_tickets(new_tickets, embeddings)
        
        if success:
            print("Successfully seeded Pinecone!")
        else:
            print("Failed to seed Pinecone.")
            
    except Exception as e:
        print(f"Error during Pinecone sync: {e}")

if __name__ == "__main__":
    seed_data()
