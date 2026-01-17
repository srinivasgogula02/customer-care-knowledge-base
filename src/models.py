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

def save_new_ticket(filepath: str, new_ticket: Dict) -> bool:
    """
    Appends a new ticket to the JSON file.
    
    Args:
        filepath: Path to the JSON file.
        new_ticket: Dictionary containing the new ticket data.
        
    Returns:
        True if successful, False otherwise.
    """
    # Validate ticket has required fields
    if not new_ticket or not new_ticket.get('issue'):
        print("Error: Ticket must have an 'issue' field.")
        return False
        
    tickets = load_tickets(filepath)
    tickets.append(new_ticket)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(tickets, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving ticket: {e}")
        return False
