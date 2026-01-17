import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pinecone_service import PineconeService

def clear_index():
    print("Initializing Pinecone Service...")
    pc = PineconeService()
    
    confirm = input("Are you sure you want to delete ALL vectors? (yes/no): ")
    if confirm.lower() == 'yes':
        if pc.delete_all_vectors():
            print("Successfully cleared index.")
        else:
            print("Failed to clear index.")
    else:
        print("Operation cancelled.")

if __name__ == "__main__":
    clear_index()
