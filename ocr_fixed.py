"""
Fixed OCR implementation for PDF text extraction
"""

import fitz  # PyMuPDF
import os
import io
import base64
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

def extract_text_from_pdf(pdf_path: str, max_pages: int = 5):
    """Extract text from PDF using OCR with OpenAI Vision API"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY in your .env file")
        return None

    print(f"📖 Processing PDF with OCR (first {max_pages} pages)...")
    document = fitz.open(pdf_path)

    all_text = ""

    for page_num in range(min(max_pages, len(document))):
        print(f"\n  Processing page {page_num + 1}...")

        try:
            page = document[page_num]

            # Convert page to image at higher resolution
            mat = fitz.Matrix(3.0, 3.0)  # 3x zoom for better OCR
            pix = page.get_pixmap(matrix=mat)

            # Get image bytes
            img_data = pix.tobytes("png")
            base64_image = base64.b64encode(img_data).decode('utf-8')

            # Call OpenAI Vision API
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
                                "text": """Extract all the text from this page of an English textbook.
                                Please provide the text in a clean, readable format.
                                If there are headings, titles, or special formatting, please indicate them clearly.
                                This is for classes 9-10 English textbook "English for Today"."""
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

            print("    Sending to OpenAI Vision API...")
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                text = response.json()['choices'][0]['message']['content']
                all_text += f"\n\n=== PAGE {page_num + 1} ===\n\n{text}\n"
                print(f"    ✅ Extracted {len(text)} characters")
            else:
                print(f"    ❌ API Error: {response.status_code}")
                print(f"    Response: {response.text[:200]}...")

        except Exception as e:
            print(f"    ❌ Error processing page: {e}")

    document.close()
    return all_text


def demo_qa(text: str):
    """Demo Q&A with the extracted text"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return

    print("\n🤖 Demo Question Answering")

    questions = [
        "What is this textbook about?",
        "What topics are covered in these pages?",
        "What can you tell me about the content structure?"
    ]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    for question in questions:
        print(f"\n📝 Q: {question}")

        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant analyzing an English textbook for classes 9-10. Answer based only on the provided text."
                },
                {
                    "role": "user",
                    "content": f"Text from textbook:\n{text[:4000]}\n\nQuestion: {question}\n\nAnswer:"
                }
            ],
            "max_tokens": 300,
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
                answer = response.json()['choices'][0]['message']['content']
                print(f"💡 A: {answer}")
            else:
                print(f"❌ Error: {response.status_code}")

        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    pdf_path = "Secondary - 2018 - Class - 9&10 - English for today-9-10  PDF Web .pdf"

    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return

    # Extract text
    extracted_text = extract_text_from_pdf(pdf_path, max_pages=3)

    if extracted_text:
        # Save to file
        with open("extracted_text.txt", "w", encoding="utf-8") as f:
            f.write(extracted_text)
        print("\n✅ Saved extracted text to 'extracted_text.txt'")

        # Show sample
        print("\n📝 Sample extracted text:")
        print(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)

        # Demo Q&A
        demo_qa(extracted_text)
    else:
        print("\n❌ No text was extracted")


if __name__ == "__main__":
    main()