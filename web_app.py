"""
Simple Web Interface for PDF Q&A
"""

import streamlit as st
import os
import fitz
import base64
import requests
from dotenv import load_dotenv
import io

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

# Sidebar
st.sidebar.title("⚙️ Settings")

# Page selection
pdf_path = "Secondary - 2018 - Class - 9&10 - English for today-9-10  PDF Web .pdf"
document = fitz.open(pdf_path)
total_pages = len(document)

selected_page = st.sidebar.slider(
    "Select page to view",
    min_value=1,
    max_value=total_pages,
    value=1
)

# Display page preview
st.sidebar.subheader(f"📄 Page {selected_page} Preview")
try:
    page = document[selected_page - 1]
    mat = fitz.Matrix(1.5, 1.5)
    pix = page.get_pixmap(matrix=pix)
    img_data = pix.tobytes("png")
    st.sidebar.image(img_data, caption=f"Page {selected_page}")
except Exception as e:
    st.sidebar.error(f"Error loading page: {e}")

document.close()

# Main content
tab1, tab2 = st.tabs(["❓ Ask Question", "📖 Extract Text"])

with tab1:
    st.subheader("Ask a Question")

    # Question input
    question = st.text_input(
        "What would you like to know about the textbook?",
        placeholder="e.g., What is this textbook about?",
        key="question_input"
    )

    # Number of pages to analyze
    num_pages = st.slider("Number of pages to analyze", min_value=1, max_value=5, value=2)

    if question:
        with st.spinner("🔍 Analyzing PDF and generating answer..."):
            try:
                # Extract text from selected pages
                api_key = os.getenv("OPENAI_API_KEY")
                all_text = ""

                for i in range(min(num_pages, total_pages)):
                    page_num = i + 1
                    page = fitz.open(pdf_path)[page_num - 1]

                    # Convert to image
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    base64_image = base64.b64encode(img_data).decode('utf-8')

                    # Call Vision API
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
                                        "text": "Extract all text from this page. Provide it in a clean, readable format."
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
                        "max_tokens": 1500
                    }

                    response = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=30
                    )

                    if response.status_code == 200:
                        text = response.json()['choices'][0]['message']['content']
                        all_text += f"\n\n=== PAGE {page_num} ===\n{text}\n"

                # Generate answer
                if all_text:
                    payload = {
                        "model": "gpt-3.5-turbo",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful assistant answering questions about an English textbook for classes 9-10. Use only the provided text to answer."
                            },
                            {
                                "role": "user",
                                "content": f"Text from textbook:\n{all_text[:6000]}\n\nQuestion: {question}\n\nProvide a helpful answer based on the text:"
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
                        answer = response.json()['choices'][0]['message']['content']

                        # Display answer
                        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                        st.subheader("💡 Answer")
                        st.write(answer)
                        st.markdown('</div>', unsafe_allow_html=True)

                        # Show extracted text
                        with st.expander("📄 View extracted text"):
                            st.text_area("Extracted Content:", all_text, height=300)
                    else:
                        st.error(f"Error generating answer: {response.status_code}")

            except Exception as e:
                st.error(f"Error: {str(e)}")

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
            try:
                page = fitz.open(pdf_path)[extract_page - 1]

                # Display page image
                mat = fitz.Matrix(1.5, 1.5)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                st.image(img_data, caption=f"Page {extract_page}")

                # Extract text with OCR
                api_key = os.getenv("OPENAI_API_KEY")
                base64_image = base64.b64encode(img_data).decode('utf-8')

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
                                    "text": """Extract all text from this page of the English textbook.
                                    Preserve formatting, headings, and structure.
                                    If there are exercises, questions, or special sections, clearly label them."""
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
                    extracted_text = response.json()['choices'][0]['message']['content']

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
                    st.error(f"Error extracting text: {response.status_code}")

            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>📚 English Textbook Q&A System | Built with OpenAI Vision API</p>
    <p>Processing image-based PDF using OCR for text extraction</p>
</div>
""", unsafe_allow_html=True)