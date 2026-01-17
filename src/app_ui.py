import streamlit as st

def render_header():
    st.title("Customer Care Knowledge Base Assistant")
    st.markdown("""
    This tool assists agents by finding similar past resolved tickets and generating proposed responses.
    """)

def render_input_area():
    with st.form("ticket_form"):
        customer_issue = st.text_area("Paste Customer Issue/Mail Content:", height=150)
        submitted = st.form_submit_button("Analyze Issue")
    return customer_issue, submitted

def render_similar_tickets(results):
    st.subheader("Traceback: Similar Resolved Tickets")
    if not results:
        st.info("No similar tickets found.")
        return

    for item in results:
        ticket = item['ticket']
        score = item['score']
        
        with st.expander(f"Similarity: {score:.2f} - {ticket['issue'][:50]}..."):
            st.markdown(f"**Category:** {ticket.get('category', 'N/A')}")
            st.markdown(f"**Issue:** {ticket['issue']}")
            st.markdown(f"**Resolution:** {ticket['resolution']}")

def render_suggested_response(response_text):
    st.subheader("Proposed Response (Generated)")
    st.success(response_text)
    
    if st.button("Copy to Clipboard (Mock)"):
        st.toast("Response copied to clipboard!")

def render_contribute_section():
    st.header("Contribute to Knowledge Base")
    st.markdown("Add a new resolved ticket to help improve future suggestions.")
    
    category = st.selectbox("Category", [
        "Other Issues",
        "General HR Support Issues",
        "Taxation & Investment Issues",
        "Letters & Documents Issues",
        "HR Systems & Portals Issues",
        "Payroll & Compensation Issues",
        "Employee Records Issues",
        "Leave Management Issues",
        "Attendance Issues"
    ])
            
    issue_raw = st.text_area("Issue Description", height=100)
    resolution_raw = st.text_area("Resolution Steps", height=100)
        
    return category, issue_raw, resolution_raw, None

