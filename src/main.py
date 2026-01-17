import streamlit as st
import os
import sys

# Add the project root to python path to allow imports if running from src or root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.models import load_tickets, save_new_ticket
from src.logic import generate_embeddings, find_similar_tickets, generate_answer, polish_ticket_content, process_pdf_text, extract_tickets_from_text
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

    # Create a placeholder for messages so the component tree remains stable
    msg_container = st.empty()

    # Check for success message from previous run (e.g. after PDF upload)
    if 'upload_status' in st.session_state and st.session_state.upload_status:
        msg_container.success(st.session_state.upload_status)
        # Clear it so it doesn't show on subsequent unrelated reruns
        del st.session_state['upload_status']
    
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
        st.subheader("Add Single Ticket")
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

        st.divider()
        st.subheader("Upload PDF Document")
        st.info("Upload a PDF to automatically extract support Q&A pairs.")
        
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            if st.button("Process PDF"):
                with st.spinner("Extracting content from PDF (this may take a moment)..."):
                    # 1. Extract text
                    text = process_pdf_text(uploaded_file)
                    if not text:
                        st.error("Could not extract text from PDF.")
                    else:
                        # 2. Parse with Groq
                        extracted_tickets = extract_tickets_from_text(text)
                        
                        if not extracted_tickets:
                            st.warning("No specific Q&A pairs found in the document.")
                        else:
                            st.success(f"Found {len(extracted_tickets)} potential items.")
                            
                            # 3. Save loop
                            success_count = 0
                            progress_bar = st.progress(0)
                            
                            topic = uploaded_file.name
                            
                            # Prepare for batch Pinecone update to save API calls/time if possible
                            # But current logic.py/main.py does one by one. 
                            # We will just do a loop for now as per "get it working" mode.
                            
                            batch_tickets_for_pinecone = []
                            batch_embeddings = []
                            
                            for i, item in enumerate(extracted_tickets):
                                new_ticket = {
                                    "category": "Document Upload",
                                    "issue": item.get('issue'),
                                    "resolution": item.get('resolution'),
                                    "source": topic
                                }
                                
                                # Save locally
                                if save_new_ticket(DATA_PATH, new_ticket):
                                    # Generate embedding
                                    # We can optimize this by batching embeddings too, but logic.py `generate_embeddings` takes a list.
                                    # Let's collect them.
                                    batch_tickets_for_pinecone.append(new_ticket)
                                    success_count += 1
                                
                                progress_bar.progress((i + 1) / len(extracted_tickets))
                                
                            if batch_tickets_for_pinecone:
                                with st.spinner("Generating embeddings and syncing to Pinecone..."):
                                    # Generate all embeddings at once
                                    corpus = [t['issue'] for t in batch_tickets_for_pinecone]
                                    embeddings_batch = generate_embeddings(corpus)
                                    
                                    # Upload to Pinecone
                                    try:
                                        PineconeService().upsert_tickets(batch_tickets_for_pinecone, embeddings_batch)
                                        # Use session state to persist message across rerun
                                        st.session_state.upload_status = f"Successfully added {success_count} content segments from '{topic}'!"
                                        st.cache_resource.clear()
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Saved locally but failed to sync to Pinecone: {e}")
                            else:
                                if success_count == 0:
                                    st.error("Failed to save extracted segments.")

if __name__ == "__main__":
    main()
