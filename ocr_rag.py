"""
RAG System with OCR for Image-based PDFs
"""

import fitz  # PyMuPDF
import os
import json
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
import base64
import requests

# Load environment variables
load_dotenv()

class OCRRAG:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.chunks = []
        self.embeddings = []

        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Please set OPENAI_API_KEY in your .env file")
        self.api_key = os.getenv("OPENAI_API_KEY")

    def extract_text_with_ocr(self, max_pages: int = 20):
        """Extract text using OCR on first few pages"""
        print(f"📖 Processing PDF with OCR (first {max_pages} pages)...")
        document = fitz.open(self.pdf_path)

        all_text = ""
        total_pages = min(max_pages, len(document))

        for page_num in range(total_pages):
            print(f"  Processing page {page_num + 1}/{total_pages}...")

            page = document[page_num]

            # Convert page to image
            mat = fitz.Matrix(2.0, 2.0)  # zoom for better OCR
            pix = page.get_pixmap(matrix=mat)

            # Save image temporarily
            img_path = f"temp_page_{page_num}.png"
            pix.save(img_path)

            # Use OpenAI's Vision API to extract text
            try:
                with open(img_path, "r") as image_file:
                    # Read and encode image
                    image_data = image_file.read()
                    base64_image = base64.b64encode(image_data).decode('utf-8')

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }

                payload = {
                    "model": "gpt-4-vision-preview",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Extract all text from this textbook page. Preserve the structure and formatting as much as possible."
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
                    json=payload
                )

                if response.status_code == 200:
                    text = response.json()['choices'][0]['message']['content']
                    all_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"
                    print(f"    Extracted {len(text)} characters")
                else:
                    print(f"    Error: {response.status_code}")

            except Exception as e:
                print(f"    OCR error: {e}")

            # Clean up temp file
            try:
                os.remove(img_path)
            except:
                pass

        document.close()
        print(f"\n✅ Total extracted text: {len(all_text)} characters")
        return all_text

    def create_chunks(self, text: str):
        """Create text chunks"""
        print("\n✂️ Creating chunks...")
        chunk_size = 1000
        chunk_overlap = 200

        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 100:  # Only keep meaningful chunks
                self.chunks.append({
                    "id": len(self.chunks),
                    "text": chunk,
                    "start_char": i
                })

        print(f"✅ Created {len(self.chunks)} chunks")

    def ask_question_simple(self, question: str, context: str) -> str:
        """Ask a question about the context"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant answering questions about an English textbook. Use only the provided context to answer."
                },
                {
                    "role": "user",
                    "content": f"Context from the textbook:\n{context[:3000]}\n\nQuestion: {question}\n\nAnswer:"
                }
            ],
            "max_tokens": 500,
            "temperature": 0
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code} - {response.text}"

    def run_demo(self):
        """Run a demo with limited pages"""
        print("\n🚀 Starting OCR RAG Demo")
        print("Note: Processing only first 10 pages for demo\n")

        # Extract text with OCR
        extracted_text = self.extract_text_with_ocr(max_pages=10)

        if not extracted_text:
            print("❌ No text extracted. Please check the PDF and API key.")
            return

        # Create chunks
        self.create_chunks(extracted_text)

        # Save extracted text
        with open("extracted_text.txt", "w") as f:
            f.write(extracted_text)
        print("\n✅ Saved extracted text to 'extracted_text.txt'")

        # Demo questions
        print("\n🤖 Demo Questions:")

        demo_questions = [
            "What topics are covered in this textbook?",
            "What is mentioned about grammar?",
            "What kinds of reading materials are included?"
        ]

        for question in demo_questions:
            print(f"\n📝 Q: {question}")
            answer = self.ask_question_simple(question, extracted_text)
            print(f"💡 A: {answer}")


if __name__ == "__main__":
    # Check if API key exists
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY in your .env file")
        print("   Create a .env file with: OPENAI_API_KEY=your_key_here")
        exit(1)

    # Run the demo
    rag = OCRRAG("Secondary - 2018 - Class - 9&10 - English for today-9-10  PDF Web .pdf")
    rag.run_demo()