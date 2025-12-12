"""
Process the full textbook (all 209 pages)
This will take time but provide complete coverage
"""

import os
import fitz
import base64
import requests
import pickle
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
import time
import random

load_dotenv()

class FullBookProcessor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.chunks = []
        self.embeddings = []
        self.metadata = []

    def process_all_pages(self):
        """Process all 209 pages"""
        print("📖 Processing ALL 209 pages of the textbook...")
        print("⚠️  This will take approximately 30-60 minutes and use significant API credits")

        document = fitz.open(self.pdf_path)
        total_pages = len(document)
        print(f"Total pages in PDF: {total_pages}")

        # Process all pages
        for page_num in range(1, total_pages + 1):
            print(f"\nProcessing page {page_num}/{total_pages}...")

            try:
                page = document[page_num - 1]

                # Convert to image
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                base64_image = base64.b64encode(img_data).decode('utf-8')

                # Extract text
                text = self._extract_text_with_vision(base64_image, page_num)

                if text and len(text) > 50:
                    # Create metadata
                    metadata = {
                        "page": page_num,
                        "char_count": len(text),
                        "processed_at": datetime.now().isoformat()
                    }

                    self.chunks.append(text)
                    self.metadata.append(metadata)
                    print(f"    ✅ Extracted {len(text)} characters")
                else:
                    print(f"    - No text extracted")

                # Rate limiting to avoid API limits
                if page_num % 10 == 0:
                    print(f"    Pausing for 30 seconds to avoid rate limits...")
                    time.sleep(30)

            except Exception as e:
                print(f"    ❌ Error: {e}")
                continue

        document.close()
        print(f"\n✅ Total pages processed: {len(self.chunks)}")
        print(f"   Total characters: {sum(len(c) for c in self.chunks):,}")
        return len(self.chunks) > 0

    def _extract_text_with_vision(self, base64_image: str, page_num: int):
        """Extract text using OpenAI Vision API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Adjust prompt based on page number
        if page_num <= 10:
            prompt = "Extract ALL text from this page. This appears to be early pages (contents, preface, etc). Include any unit titles, lesson names, or table of contents."
        elif page_num <= 100:
            prompt = """Extract ALL text from this textbook page. Focus on:
            - Unit numbers and titles
            - Lesson titles
            - Reading passages
            - Exercise instructions
            - Questions and activities
            - Grammar explanations
            Include all text content thoroughly."""
        else:
            prompt = """Extract ALL text from this page of the English textbook. Include:
            - All reading passages
            - Exercises and activities
            - Grammar points
            - Instructions
            - Any text content on the page
            Be thorough and complete."""

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
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
            "max_tokens": 3000
        }

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                print(f"    API Error: {response.status_code}")
                return None

        except Exception as e:
            print(f"    OCR Error: {e}")
            return None

    def create_embeddings_batch(self):
        """Create embeddings for all chunks"""
        print(f"\n🧠 Creating embeddings for {len(self.chunks)} chunks...")
        print("This will take approximately 30-60 minutes...")

        batch_size = 5  # Process 5 at a time
        errors = 0

        for i in range(0, len(self.chunks), batch_size):
            batch_end = min(i + batch_size, len(self.chunks))
            print(f"  Processing batch {i//batch_size + 1}/{(len(self.chunks)//batch_size)+1} (chunks {i+1}-{batch_end})")

            for j in range(i, batch_end):
                try:
                    embedding = self._create_single_embedding(self.chunks[j])
                    if embedding:
                        self.embeddings.append(embedding)
                        print(f"    ✅ Embedded chunk {j+1}/{len(self.chunks)}")
                    else:
                        # Add empty embedding to maintain index
                        self.embeddings.append([0] * 1536)
                        errors += 1
                        print(f"    ⚠️  Failed to embed chunk {j+1}")

                except Exception as e:
                    print(f"    ❌ Error embedding chunk {j+1}: {e}")
                    self.embeddings.append([0] * 1536)
                    errors += 1

                # Small delay between embeddings
                time.sleep(0.1)

            # Pause between batches
            if i + batch_size < len(self.chunks):
                print(f"    Pausing 5 seconds between batches...")
                time.sleep(5)

        print(f"\n✅ Created {len(self.embeddings)} embeddings (with {errors} errors)")

    def _create_single_embedding(self, text: str):
        """Create embedding for a single text chunk"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Use first 8000 characters
        text = text[:8000]

        payload = {
            "model": "text-embedding-ada-002",
            "input": text
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
                return None

        except Exception as e:
            return None

    def save_database(self):
        """Save the complete vector database"""
        print(f"\n💾 Saving complete vector database...")

        db_data = {
            "chunks": self.chunks,
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "created_at": datetime.now().isoformat(),
            "total_chunks": len(self.chunks),
            "total_pages": len(set(m['page'] for m in self.metadata)),
            "total_characters": sum(len(c) for c in self.chunks)
        }

        with open("textbook_full_209_pages.pkl", 'wb') as f:
            pickle.dump(db_data, f)

        print(f"✅ Full database saved!")
        print(f"   Chunks: {len(self.chunks):,}")
        print(f"   Pages: {len(set(m['page'] for m in self.metadata)):,}")
        print(f"   Characters: {sum(len(c) for c in self.chunks):,}")
        print(f"   File size: {os.path.getsize('textbook_full_209_pages.pkl') / 1024 / 1024:.2f} MB")

        # Also copy to vector_db.pkl for compatibility
        import shutil
        shutil.copy("textbook_full_209_pages.pkl", "vector_db.pkl")
        print("✅ Also saved as vector_db.pkl for compatibility")


def main():
    print("=" * 60)
    print("📚 ENGLISH TEXTBOOK FULL PROCESSOR")
    print("Processing ALL 209 pages")
    print("=" * 60)

    # Cost estimate
    estimated_cost = 209 * 0.01  # Approximately $0.01 per page for Vision API
    embedding_cost = len(open("textbook_vector_db.pkl", "rb").read()) / 1024 / 1024 * 0.0001  # Rough estimate
    print(f"\n💰 Estimated cost: ${estimated_cost:.2f} for OCR + ${embedding_cost:.2f} for embeddings")
    print(f"Total estimated: ${estimated_cost + embedding_cost:.2f}")

    confirm = input("\n⚠️  This will process all 209 pages and cost significant API credits. Continue? (y/n): ")

    if confirm.lower() != 'y':
        print("Cancelled.")
        return

    # Process the book
    processor = FullBookProcessor("Secondary - 2018 - Class - 9&10 - English for today-9-10  PDF Web .pdf")

    # Process all pages
    if processor.process_all_pages():
        # Create embeddings
        processor.create_embeddings_batch()

        # Save database
        processor.save_database()

        print("\n✅ Full textbook processing complete!")
        print("Now restart the RAG app to use the complete database.")
    else:
        print("\n❌ Processing failed")


if __name__ == "__main__":
    main()