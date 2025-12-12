"""
RAG Web App with Vector Database
Fast querying using pre-built embeddings
"""

import streamlit as st
import os
import pickle
import numpy as np
import requests
from dotenv import load_dotenv
from datetime import datetime

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
    .stats-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📚 English Textbook RAG System</h1>
    <p>Fast Q&A using Vector Database Embeddings</p>
</div>
""", unsafe_allow_html=True)

# Load vector database
@st.cache_resource
def load_vector_database():
    """Load the vector database"""
    # Get absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "vector_db.pkl")

    
    if not os.path.exists(db_path):
        st.error("❌ Vector database not found!")
        st.error(f"Looking for: {db_path}")
        st.info("Please run `python3 full_rag_system.py` first to create the database.")
        return None

    try:
        with open(db_path, 'rb') as f:
            db_data = pickle.load(f)

        st.session_state['db'] = db_data
        return db_data
    except Exception as e:
        st.error(f"❌ Error loading database: {e}")
        return None

# Initialize session state
if 'db' not in st.session_state:
    with st.spinner("Loading vector database..."):
        db_data = load_vector_database()
        if db_data:
            st.success("✅ Vector database loaded!")
            st.session_state['query_count'] = 0

def create_embedding(text):
    """Create embedding for query text"""
    api_key = os.getenv("OPENAI_API_KEY")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "text-embedding-ada-002",
        "input": text[:8000]
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()['data'][0]['embedding']
        else:
            st.error(f"Embedding API Error: {response.status_code}")
            return None

    except Exception as e:
        st.error(f"Embedding Error: {e}")
        return None

def search_similar(query, k=3):
    """Search for similar chunks in vector database"""
    if 'db' not in st.session_state:
        return []

    db = st.session_state['db']
    chunks = db['chunks']
    embeddings = db['embeddings']
    metadata = db['metadata']

    # Create embedding for query
    query_embedding = create_embedding(query)
    if not query_embedding:
        return []

    query_embedding = np.array(query_embedding)

    # Calculate similarities
    similarities = []
    for i, embedding in enumerate(embeddings):
        embedding_array = np.array(embedding)
        # Skip empty embeddings
        if np.sum(embedding_array) == 0:
            continue
        similarity = np.dot(query_embedding, embedding_array)
        similarities.append((i, similarity))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top k results
    results = []
    for i, similarity in similarities[:k]:
        results.append({
            "chunk": chunks[i],
            "metadata": metadata[i],
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
                "content": """You are a helpful assistant for the English textbook "English for Today" (Classes 9-10).
                Answer questions based ONLY on the provided context. Be precise, helpful, and concise."""
            },
            {
                "role": "user",
                "content": f"Context from textbook:\n{context[:12000]}\n\nQuestion: {question}\n\nAnswer:"
            }
        ],
        "max_tokens": 500,
        "temperature": 0
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error generating answer: {response.status_code}"

    except Exception as e:
        return f"Error: {str(e)}"

def ask_question(question):
    """Main question answering function"""
    # Search for relevant chunks
    with st.spinner("🔍 Searching vector database..."):
        relevant_chunks = search_similar(question, k=3)

    if not relevant_chunks:
        return {
            "answer": "I couldn't find relevant information in the textbook.",
            "sources": []
        }

    # Create context
    context = ""
    for i, result in enumerate(relevant_chunks):
        context += f"\n\nSource {i+1} (Page {result['metadata']['page']}):\n{result['chunk'][:1000]}...\n"

    # Generate answer
    with st.spinner("🤖 Generating answer..."):
        answer = generate_answer(question, context)

    return {
        "answer": answer,
        "sources": relevant_chunks
    }

# Sidebar
st.sidebar.title("📊 Database Info")

if 'db' in st.session_state:
    db = st.session_state['db']

    # Database statistics
    st.sidebar.subheader("Statistics")
    total_chars = sum(len(c) for c in db['chunks'])
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "vector_db.pkl")
    db_size = os.path.getsize(db_path) / 1024 / 1024 if os.path.exists(db_path) else 0
    st.sidebar.markdown(f"""
    <div class="stats-card">
        <strong>Total Chunks:</strong> {len(db['chunks']):,}<br>
        <strong>Total Characters:</strong> {total_chars:,}<br>
        <strong>Avg Chunk Size:</strong> {np.mean([len(c) for c in db['chunks']]):.0f} chars<br>
        <strong>Pages Processed:</strong> {len(set(m['page'] for m in db['metadata']))}<br>
        <strong>Created:</strong> {db.get('created_at', 'Unknown')[:10]}<br>
        <strong>Database Size:</strong> {db_size:.2f} MB
    </div>
    """, unsafe_allow_html=True)

    # Query counter
    if 'query_count' not in st.session_state:
        st.session_state['query_count'] = 0
    st.sidebar.metric("Queries Processed", st.session_state['query_count'])

# Main content
if 'db' not in st.session_state:
    st.error("Please load the vector database first.")
else:
    # Question input
    st.subheader("❓ Ask a Question")

    # Example questions
    with st.expander("💡 Example Questions"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("What is Unit 1 about?"):
                st.session_state['question_input'] = "What is Unit 1 about?"
            if st.button("What topics are covered?"):
                st.session_state['question_input'] = "What topics are covered in this textbook?"
        with col2:
            if st.button("Tell me about Lesson 1"):
                st.session_state['question_input'] = "Tell me about Lesson 1"
            if st.button("What is Good Citizens about?"):
                st.session_state['question_input'] = "What is the unit Good Citizens about?"

    # Question input field
    question = st.text_input(
        "Enter your question:",
        value=st.session_state.get('question_input', ''),
        key="question_input_main"
    )

    # Search settings
    col1, col2 = st.columns([3, 1])
    with col2:
        k = st.selectbox("Results", options=[1, 2, 3, 4, 5], index=2)

    if st.button("🔍 Get Answer", type="primary"):
        if question:
            # Update query counter
            st.session_state['query_count'] = st.session_state.get('query_count', 0) + 1

            # Get answer
            result = ask_question(question)

            # Display answer
            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
            st.subheader("💡 Answer")
            st.write(result['answer'])
            st.markdown('</div>', unsafe_allow_html=True)

            # Display sources
            if result['sources']:
                st.subheader("📚 Sources")
                for i, source in enumerate(result['sources'], 1):
                    with st.expander(f"Source {i} - Page {source['metadata']['page']} (Similarity: {source['similarity']:.2f})"):
                        st.write(f"**Page:** {source['metadata']['page']}")
                        st.write(f"**Length:** {len(source['chunk'])} characters")
                        st.write(f"**Processed:** {source['metadata'].get('processed_at', 'Unknown')[:10]}")
                        st.write("**Content:**")
                        st.write(source['chunk'])
            else:
                st.info("No sources found")
        else:
            st.warning("Please enter a question")

    # Database search
    st.markdown("---")
    st.subheader("🔍 Vector Search Only")

    search_query = st.text_input("Search for similar content (no AI generation):")

    if search_query and st.button("Search"):
        with st.spinner("Searching..."):
            results = search_similar(search_query, k=3)

            if results:
                st.success(f"Found {len(results)} similar chunks")

                for i, result in enumerate(results, 1):
                    st.markdown(f"<div class='source-box'>", unsafe_allow_html=True)
                    st.markdown(f"**Result {i}** - Page {result['metadata']['page']} (Similarity: {result['similarity']:.3f})")
                    st.write(result['chunk'][:500] + "..." if len(result['chunk']) > 500 else result['chunk'])
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("No similar content found")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📚 RAG System with Vector Database | Response time: < 1 second</p>
    <p>No re-processing needed - uses pre-computed embeddings</p>
</div>
""", unsafe_allow_html=True)