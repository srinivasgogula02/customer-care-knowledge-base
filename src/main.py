import streamlit as st
import os
import sys

# Add the project root to python path to allow imports if running from src or root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.models import load_tickets, save_new_ticket
from src.logic import generate_embeddings, find_similar_tickets, generate_answer, polish_ticket_content
import src.app_ui as ui
from src.pinecone_service import PineconeService

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
    
    # Extract text content for embedding
    corpus = [t['issue'] for t in tickets]
    embeddings = generate_embeddings(corpus)
    
    # Sync to Pinecone
    with st.spinner("Syncing to Pinecone..."):
        pc = PineconeService()
        pc.upsert_tickets(tickets, embeddings)
        
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

    tab1, tab2 = st.tabs(["Search Tickets", "Contribute"])

    with tab1:
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

    with tab2:
        category, issue_raw, resolution_raw, polish_btn = ui.render_contribute_section()
        
        if 'polished_issue' not in st.session_state:
            st.session_state.polished_issue = ""
        if 'polished_resolution' not in st.session_state:
            st.session_state.polished_resolution = ""
            
        if polish_btn:
            if issue_raw and resolution_raw:
                with st.spinner("Polishing content with AI..."):
                    p_issue, p_res = polish_ticket_content(issue_raw, resolution_raw)
                    st.session_state.polished_issue = p_issue
                    st.session_state.polished_resolution = p_res
                st.rerun()
            else:
                st.warning("Please fill in both Issue and Resolution.")
                
        if st.session_state.polished_issue:
            st.divider()
            st.subheader("Preview & Save")
            final_issue = st.text_area("Polished Issue", value=st.session_state.polished_issue, height=100)
            final_resolution = st.text_area("Polished Resolution", value=st.session_state.polished_resolution, height=100)
            
            if st.button("Save to Knowledge Base"):
                new_ticket = {
                    "category": category,
                    "issue": final_issue,
                    "resolution": final_resolution
                }
                if save_new_ticket(DATA_PATH, new_ticket):
                    # Sync to Pinecone
                    try:
                        new_embedding = generate_embeddings([new_ticket['issue']])
                        PineconeService().upsert_tickets([new_ticket], new_embedding)
                    except Exception as e:
                        st.error(f"Saved locally but failed to sync to Pinecone: {e}")
                        
                    st.success("Ticket saved successfully!")
                    # Clear state
                    st.session_state.polished_issue = ""
                    st.session_state.polished_resolution = ""
                    # Clear cache to reload new data
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error("Failed to save ticket.")

if __name__ == "__main__":
    main()
