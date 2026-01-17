import sys
import os

# Add src to path so we can import PineconeService
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pinecone_service import PineconeService

def main():
    print("Initializing Pinecone Service...")
    service = PineconeService()
    
    print("Clearing all vectors from index...")
    success = service.delete_all_vectors()
    
    if success:
        print("Successfully cleared all vectors.")
    else:
        print("Failed to clear vectors.")

if __name__ == "__main__":
    main()
