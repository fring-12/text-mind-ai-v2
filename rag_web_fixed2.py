"""
Fixed RAG Web App - Simplified Version
"""

import streamlit as st
import os
import pickle
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ Please set OPENAI_API_KEY in your .env file")
    st.stop()

# Configure page
st.set_page_config(
    page_title="English Textbook RAG System",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .source-box {
        background-color: #fff;
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📚 English Textbook RAG System</h1>
    <p>Ask questions about "English for Today" (Classes 9-10)</p>
</div>
""", unsafe_allow_html=True)

# Load database function
@st.cache_resource
def load_database():
    """Load the vector database"""
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vector_db.pkl")

    try:
        with open(db_path, 'rb') as f:
            db = pickle.load(f)
        return db
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return None

# Load database at startup
db = load_database()

if db is None:
    st.error("❌ Failed to load vector database. Please check the file.")
    st.stop()

# Show database info
st.sidebar.title("📊 Database Info")
st.sidebar.write(f"Chunks loaded: {len(db.get('chunks', []))}")
st.sidebar.write(f"Embeddings: {len(db.get('embeddings', []))}")
st.sidebar.write(f"Database size: {os.path.getsize(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vector_db.pkl')) / 1024 / 1024:.2f} MB")

# RAG functions
def create_embedding(text):
    """Create embedding for query"""
    api_key = os.getenv("OPENAI_API_KEY")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "text-embedding-ada-002",
        "input": text[:8000]
    }

    response = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers=headers,
        json=payload,
        timeout=30
    )

    if response.status_code == 200:
        return response.json()['data'][0]['embedding']
    return None

def search_similar(query, k=3):
    """Search for similar chunks"""
    query_embedding = create_embedding(query)
    if not query_embedding:
        return []

    query_embedding = np.array(query_embedding)

    similarities = []
    for i, embedding in enumerate(db['embeddings']):
        embedding_array = np.array(embedding)
        if np.sum(embedding_array) == 0:
            continue
        similarity = np.dot(query_embedding, embedding_array)
        similarities.append((i, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)

    results = []
    for i, similarity in similarities[:k]:
        results.append({
            "chunk": db['chunks'][i],
            "metadata": db['metadata'][i],
            "similarity": float(similarity)
        })

    return results

def generate_answer(question, context):
    """Generate answer using GPT"""
    api_key = os.getenv("OPENAI_API_KEY")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant for the English textbook 'English for Today' (Classes 9-10). Answer questions based ONLY on the provided context."
            },
            {
                "role": "user",
                "content": f"Context: {context[:12000]}\n\nQuestion: {question}\n\nAnswer:"
            }
        ],
        "max_tokens": 500,
        "temperature": 0
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30
    )

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    return f"Error: {response.status_code}"

# Main interface
st.subheader("❓ Ask a Question")

# Example questions
with st.expander("💡 Example Questions"):
    if st.button("What is Unit 1 about?"):
        st.session_state['question'] = "What is Unit 1 about?"
    if st.button("Tell me about Good Citizens"):
        st.session_state['question'] = "Tell me about Good Citizens"

# Question input
question = st.text_input(
    "Enter your question:",
    value=st.session_state.get('question', ''),
    key="question_input"
)

if st.button("🔍 Get Answer", type="primary"):
    if question:
        with st.spinner("🔍 Searching..."):
            # Search
            results = search_similar(question, k=3)

            if results:
                # Create context
                context = ""
                for i, result in enumerate(results):
                    context += f"\n\nSource {i+1} (Page {result['metadata']['page']}):\n{result['chunk'][:1000]}..."

                # Generate answer
                with st.spinner("🤖 Generating answer..."):
                    answer = generate_answer(question, context)

                # Display answer
                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                st.subheader("💡 Answer")
                st.write(answer)
                st.markdown('</div>', unsafe_allow_html=True)

                # Display sources
                if results:
                    st.subheader("📚 Sources")
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Source {i} - Page {result['metadata']['page']}"):
                            st.write(result['chunk'])
            else:
                st.warning("No relevant information found")
    else:
        st.warning("Please enter a question")

# Quick test
st.markdown("---")
st.subheader("🧪 Quick Test")

# Show first few chunks
if db.get('chunks'):
    st.write("Sample content from database:")
    for i, chunk in enumerate(db['chunks'][:3]):
        with st.expander(f"Chunk {i+1} (Page {db['metadata'][i]['page']})"):
            st.write(chunk[:300] + "...")