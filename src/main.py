import streamlit as st
import os
import sys

# Add the project root to python path to allow imports if running from src or root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.models import load_tickets
from src.logic import generate_embeddings, find_similar_tickets, generate_answer
import src.app_ui as ui

# Configuration
DATA_PATH = os.path.join(parent_dir, 'data', 'tickets.json')

@st.cache_resource
def initialize_knowledge_base():
    """
    Loads tickets and pre-computes embeddings.
    Cached so it only runs once.
    """
    tickets = load_tickets(DATA_PATH)
    if not tickets:
        return [], None
    
    # Extract text content for embedding (combining issue and resolution for better context or just issue)
    # Usually searching by issue description is best for finding similar problems.
    corpus = [t['issue'] for t in tickets]
    embeddings = generate_embeddings(corpus)
    return tickets, embeddings

def main():
    st.set_page_config(page_title="Customer Care KB", layout="wide")
    
    ui.render_header()
    
    # Initialize KB
    with st.spinner("Loading Knowledge Base..."):
        tickets, embeddings = initialize_knowledge_base()
        
    if not tickets:
        st.error("Failed to load knowledge base data. Please check data/tickets.json")
        return

    # Input
    query, submitted = ui.render_input_area()
    
    if submitted and query:
        col1, col2 = st.columns(2)
        
        # Search Step
        with st.spinner("Searching for similar tickets..."):
            similar_results = find_similar_tickets(query, tickets, embeddings)
        
        with col1:
            ui.render_similar_tickets(similar_results)
            
        # Generation Step
        with col2:
            with st.spinner("Generating response..."):
                answer = generate_answer(query, similar_results)
            ui.render_suggested_response(answer)
    elif submitted and not query:
        st.warning("Please enter an issue description.")

if __name__ == "__main__":
    main()
