"""
Fixed Web Interface for PDF Q&A
Handles PDF access properly to avoid orphaned object errors
"""

import streamlit as st
import os
import fitz
import base64
import requests
from dotenv import load_dotenv
import io
import gc

load_dotenv()

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ Please set OPENAI_API_KEY in your .env file")
    st.stop()

# Configure page
st.set_page_config(
    page_title="English Textbook Q&A",
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
    }
    .page-preview {
        background-color: #fff;
        border: 1px solid #ddd;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📚 English Textbook Q&A</h1>
    <p>Ask questions about "English for Today" (Classes 9-10)</p>
</div>
""", unsafe_allow_html=True)

# Helper functions
def get_page_image(page_num: int):
    """Get image of a specific page"""
    try:
        pdf_path = "Secondary - 2018 - Class - 9&10 - English for today-9-10  PDF Web .pdf"
        document = fitz.open(pdf_path)
        page = document[page_num - 1]
        mat = fitz.Matrix(1.5, 1.5)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        document.close()
        return img_data
    except Exception as e:
        st.error(f"Error loading page: {e}")
        return None

def extract_text_with_ocr(page_num: int):
    """Extract text from a specific page using OCR"""
    try:
        img_data = get_page_image(page_num)
        if img_data is None:
            return None

        base64_image = base64.b64encode(img_data).decode('utf-8')
        api_key = os.getenv("OPENAI_API_KEY")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Extract ALL text from this page of the English textbook.
                            Pay special attention to:
                            - Unit numbers and titles (e.g., "Unit 1: People")
                            - Lesson titles
                            - Headings and subheadings
                            - Exercise instructions
                            - Any questions or activities

                            Please format the output clearly with headings marked."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2000
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            st.error(f"OCR Error: {response.status_code}")
            return None

    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return None

def ask_question_with_context(question: str, context: str):
    """Ask a question with provided context"""
    try:
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
                    "content": """You are a helpful assistant answering questions about an English textbook for classes 9-10.
                    Use only the provided text to answer. Be precise and look for specific information like unit names,
                    lesson titles, and content structure."""
                },
                {
                    "role": "user",
                    "content": f"Text from textbook:\n{context[:8000]}\n\nQuestion: {question}\n\nAnswer based on the text:"
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
        else:
            return f"Error: {response.status_code} - {response.text}"

    except Exception as e:
        return f"Error: {str(e)}"

# Sidebar
st.sidebar.title("⚙️ Settings")

# Page selection for preview
pdf_path = "Secondary - 2018 - Class - 9&10 - English for today-9-10  PDF Web .pdf"
try:
    document = fitz.open(pdf_path)
    total_pages = len(document)
    document.close()
except:
    total_pages = 209

selected_page = st.sidebar.slider(
    "Select page to view",
    min_value=1,
    max_value=total_pages,
    value=1
)

# Display page preview in sidebar
st.sidebar.subheader(f"📄 Page {selected_page} Preview")
img_data = get_page_image(selected_page)
if img_data:
    st.sidebar.image(img_data, caption=f"Page {selected_page}")

# Main content
tab1, tab2 = st.tabs(["❓ Ask Question", "📖 Extract Text"])

with tab1:
    st.subheader("Ask a Question")

    # Example questions
    with st.expander("💡 Example Questions"):
        st.write("Try these questions:")
        st.write("• What is the name of Unit 1?")
        st.write("• What topics are covered in Unit 2?")
        st.write("• How many units are in the book?")
        st.write("• What is Lesson 1 about?")

    # Question input
    question = st.text_input(
        "What would you like to know about the textbook?",
        placeholder="e.g., What is the name of Unit 1?",
        key="question_input"
    )

    # Page range selection
    col1, col2 = st.columns(2)
    with col1:
        start_page = st.number_input("From page", min_value=1, max_value=total_pages, value=1)
    with col2:
        end_page = st.number_input("To page", min_value=1, max_value=total_pages, value=5)

    if st.button("🔍 Find Answer"):
        if question:
            with st.spinner("🔍 Analyzing PDF and generating answer..."):
                all_text = ""
                pages_to_analyze = min(end_page - start_page + 1, 10)  # Limit to 10 pages

                for i in range(pages_to_analyze):
                    page_num = start_page + i
                    st.write(f"Analyzing page {page_num}...")

                    text = extract_text_with_ocr(page_num)
                    if text:
                        all_text += f"\n\n=== PAGE {page_num} ===\n{text}\n"

                if all_text:
                    # Generate answer
                    st.write("Generating answer...")
                    answer = ask_question_with_context(question, all_text)

                    # Display answer
                    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                    st.subheader("💡 Answer")
                    st.write(answer)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Show extracted text
                    with st.expander("📄 View extracted text"):
                        st.text_area("Extracted Content:", all_text, height=300)
                else:
                    st.error("No text could be extracted from the selected pages")
        else:
            st.warning("Please enter a question")

with tab2:
    st.subheader("Extract Text from Page")

    # Page selector
    extract_page = st.number_input(
        "Enter page number to extract text from:",
        min_value=1,
        max_value=total_pages,
        value=1
    )

    if st.button("Extract Text"):
        with st.spinner("Extracting text..."):
            # Show page image
            img_data = get_page_image(extract_page)
            if img_data:
                st.image(img_data, caption=f"Page {extract_page}")

                # Extract text
                extracted_text = extract_text_with_ocr(extract_page)

                if extracted_text:
                    st.markdown('<div class="page-preview">', unsafe_allow_html=True)
                    st.subheader("📝 Extracted Text")
                    st.write(extracted_text)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Download button
                    st.download_button(
                        label="📥 Download Text",
                        data=extracted_text,
                        file_name=f"page_{extract_page}_text.txt",
                        mime="text/plain"
                    )
                else:
                    st.error("Failed to extract text from this page")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📚 English Textbook Q&A System | Built with OpenAI Vision API</p>
    <p>Processing image-based PDF using OCR for text extraction</p>
</div>
""", unsafe_allow_html=True)