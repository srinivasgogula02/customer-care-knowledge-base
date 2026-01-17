import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import save_new_ticket, load_tickets
from src.logic import polish_ticket_content

def test_add_feature():
    print("Testing 'Add Ticket' Feature...")
    
    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tickets.json')
    
    # 1. Polish
    raw_issue = "internet super slow, annoying"
    raw_res = "told him restart router, fixed it"
    print(f"Polishing:\nIssue: {raw_issue}\nRes: {raw_res}")
    
    p_issue, p_res = polish_ticket_content(raw_issue, raw_res)
    print(f"Result:\nIssue: {p_issue}\nRes: {p_res}")
    
    # 2. Save
    new_ticket = {
        "category": "Test",
        "issue": p_issue,
        "resolution": p_res
    }
    
    initial_count = len(load_tickets(DATA_PATH))
    success = save_new_ticket(DATA_PATH, new_ticket)
    
    if success:
        print("Save successful.")
        new_count = len(load_tickets(DATA_PATH))
        if new_count == initial_count + 1:
            print("PASS: Ticket count increased.")
        else:
            print("FAIL: Ticket count did not increase.")
    else:
        print("FAIL: Save returned false.")

if __name__ == "__main__":
    test_add_feature()
