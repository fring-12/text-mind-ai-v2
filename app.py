"""
Streamlit App for RAG System
Web interface for querying the English textbook RAG system
"""

import streamlit as st
import os
from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline

load_dotenv()

# Page configuration
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
    .query-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
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


def initialize_rag_system():
    """Initialize the RAG system"""
    try:
        # Get configuration from session state or environment
        embedding_model = st.session_state.get('embedding_model', 'openai')
        vector_db = st.session_state.get('vector_db', 'chroma')
        llm_provider = st.session_state.get('llm_provider', 'openai')

        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            embedding_model=embedding_model,
            vector_db_type=vector_db,
            llm_provider=llm_provider
        )

        return rag_pipeline
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None


def display_sources(sources):
    """Display source documents"""
    st.subheader("📚 Source Documents")

    for i, source in enumerate(sources):
        with st.expander(f"Source {i+1} - Page {source['metadata'].get('page', 'Unknown')}"):
            col1, col2 = st.columns([1, 3])

            with col1:
                st.markdown("**Metadata:**")
                st.write(f"- Page: {source['metadata'].get('page', 'Unknown')}")
                st.write(f"- Unit: {source['metadata'].get('unit', 'Unknown')}")
                st.write(f"- Lesson: {source['metadata'].get('lesson', 'Unknown')}")

            with col2:
                st.markdown("**Content:**")
                st.write(source['content'])


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 English Textbook RAG System</h1>
        <p>Ask questions about "English for Today" (Classes 9-10)</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for configuration
    st.sidebar.title("⚙️ Configuration")

    # Model selection
    st.sidebar.subheader("Model Settings")
    embedding_model = st.sidebar.selectbox(
        "Embedding Model",
        ["openai", "huggingface"],
        index=0 if os.getenv("EMBEDDING_MODEL", "openai") == "openai" else 1
    )
    st.session_state['embedding_model'] = embedding_model

    vector_db = st.sidebar.selectbox(
        "Vector Database",
        ["chroma", "faiss"],
        index=0 if os.getenv("VECTOR_DB", "chroma") == "chroma" else 1
    )
    st.session_state['vector_db'] = vector_db

    llm_provider = st.sidebar.selectbox(
        "LLM Provider",
        ["openai", "huggingface"],
        index=0 if os.getenv("LLM_PROVIDER", "openai") == "openai" else 1
    )
    st.session_state['llm_provider'] = llm_provider

    # Initialize RAG system
    rag_pipeline = initialize_rag_system()

    if not rag_pipeline:
        st.error("❌ Failed to initialize RAG system. Please check your configuration.")
        st.stop()

    # Main interface
    st.markdown('<div class="query-box">', unsafe_allow_html=True)

    # Query input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Ask a question about the textbook:",
            placeholder="e.g., What is the importance of learning English?",
            key="query_input"
        )
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        search_button = st.button("🔍 Search", type="primary")

    st.markdown('</div>', unsafe_allow_html=True)

    # Example questions
    with st.expander("💡 Example Questions"):
        example_queries = [
            "What is the importance of learning English?",
            "Explain the concept of narrative writing",
            "What are the different parts of speech?",
            "How do we write a formal letter?",
            "What is a simile?",
            "Describe the rules for subject-verb agreement"
        ]

        for i, example in enumerate(example_queries):
            if st.button(example, key=f"example_{i}"):
                st.session_state.query_input = example
                st.rerun()

    # Process query
    if search_button or query:
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("🔍 Searching for relevant information..."):
                try:
                    # Get number of sources to retrieve
                    k = st.sidebar.slider("Number of sources", min_value=1, max_value=10, value=4)

                    # Query the RAG system
                    result = rag_pipeline.query(query, k=k)

                    # Display answer
                    st.markdown("### 💡 Answer")
                    st.write(result['answer'])

                    # Display sources
                    if result['sources']:
                        display_sources(result['sources'])
                    else:
                        st.warning("No sources found for this query.")

                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info("""
    This RAG system allows you to ask questions about the English textbook content.

    The system:
    - Retrieves relevant passages from the textbook
    - Uses AI to generate answers based on the retrieved content
    - Shows the source pages for verification

    All answers are based on the textbook content only.
    """)


if __name__ == "__main__":
    main()