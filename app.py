import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# --- UI Setup ---
st.title("Pharma QMS Semantic Search 🧬")
st.write("Stop searching by exact keywords. Describe your proposed change below, and the AI will find the most similar historical Change Controls.")

# --- Data & Model Loading ---
# @st.cache_data tells Streamlit to only load the data once so the app doesn't freeze every time you type
@st.cache_data
def load_data():
    github_url = "https://raw.githubusercontent.com/maymoethu/qms-semantic-search/refs/heads/main/pharma_change_controls.csv"
    df = pd.read_csv(github_url)
    df['Search_Text'] = df['Title'] + ": " + df['Description']
    return df

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

df = load_data()
model = load_model()

# Convert all records to AI vectors
record_embeddings = model.encode(df['Search_Text'].tolist())

# --- The Search Bar ---
st.divider()
search_query = st.text_input("Describe your change (e.g., 'replace water loop pump'):")

# --- Search Logic ---
# If the user types something in the box, run the search
if search_query:
    query_embedding = model.encode(search_query)
    hits = util.semantic_search(query_embedding, record_embeddings, top_k=3)
    
    st.subheader("Top 3 Historical Matches:")
    
    # Loop through the results and display them nicely
    for hit in hits[0]:
        row_id = hit['corpus_id']
        match_score = hit['score']
        
        # UI formatting (putting data into visual "cards")
        st.info(f"**{df['CC_ID'][row_id]} - {df['Title'][row_id]}** (AI Confidence: {match_score:.2f})")
        st.write(f"**Description:** {df['Description'][row_id]}")
        st.caption(f"Impacted Systems: {df['Impacted_Systems'][row_id]} | Risk Level: {df['Risk_Level'][row_id]} | Department: {df['Department'][row_id]}")
        st.write("---")
