import json
import os
from typing import List, Dict

def load_tickets(filepath: str) -> List[Dict]:
    """
    Loads tickets from a JSON file.
    
    Args:
        filepath: Path to the JSON file.
        
    Returns:
        A list of dictionaries representing tickets.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return []
    
    with open(filepath, 'r') as f:
        try:
            tickets = json.load(f)
            return tickets
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from {filepath}")
            return []
