"""
Complete RAG System with Vector Database
One-time embedding and storage for fast querying
"""

import os
import fitz
import base64
import json
import numpy as np
from typing import List, Dict
import requests
from dotenv import load_dotenv
from datetime import datetime
import pickle

load_dotenv()

class FullRAGSystem:
    def __init__(self, pdf_path: str, db_path: str = "vector_db.pkl"):
        self.pdf_path = pdf_path
        self.db_path = db_path
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("Please set OPENAI_API_KEY in your .env file")

        self.chunks = []
        self.embeddings = []
        self.metadata = []

    def process_full_textbook(self, max_pages: int = 50):
        """Process and extract text from the entire textbook"""
        print(f"📖 Processing textbook (up to {max_pages} pages)...")

        document = fitz.open(self.pdf_path)
        total_pages = min(max_pages, len(document))

        # Table of contents pages (usually 5-15)
        toc_pages = list(range(5, min(15, total_pages)))
        # Content pages
        content_pages = [p for p in range(1, total_pages) if p not in toc_pages]

        all_pages = toc_pages + content_pages

        for i, page_num in enumerate(all_pages):
            print(f"  Processing page {page_num}/{total_pages} ({i+1}/{len(all_pages)})...")

            try:
                page = document[page_num - 1]

                # Convert to image with higher resolution for better OCR
                mat = fitz.Matrix(2.5, 2.5)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                base64_image = base64.b64encode(img_data).decode('utf-8')

                # Extract text with OCR
                text = self._extract_text_from_image(base64_image)

                if text and len(text) > 50:  # Only keep meaningful text
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

            except Exception as e:
                print(f"    ❌ Error: {e}")

        document.close()
        print(f"\n✅ Total pages processed: {len(self.chunks)}")
        print(f"   Total characters: {sum(len(c) for c in self.chunks):,}")

        return len(self.chunks) > 0

    def _extract_text_from_image(self, base64_image: str) -> str:
        """Extract text from a base64 encoded image"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Extract all text from this English textbook page.
                            Include:
                            - Unit numbers and titles (e.g., "Unit 1: Title")
                            - Lesson titles and numbers
                            - Exercise instructions
                            - Reading passages
                            - Questions and activities

                            Preserve the structure and format clearly."""
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
                timeout=30
            )

            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                print(f"    OCR API Error: {response.status_code}")
                return None

        except Exception as e:
            print(f"    OCR Error: {e}")
            return None

    def create_embeddings(self, batch_size: int = 5):
        """Create embeddings for all text chunks"""
        print(f"\n🧠 Creating embeddings for {len(self.chunks)} chunks...")

        for i in range(0, len(self.chunks), batch_size):
            batch_chunks = self.chunks[i:i + batch_size]

            print(f"  Processing batch {i//batch_size + 1}/{(len(self.chunks)//batch_size)+1}...")

            for chunk in batch_chunks:
                try:
                    embedding = self._create_single_embedding(chunk)
                    if embedding:
                        self.embeddings.append(embedding)
                        print(f"    ✅ Embedded chunk {len(self.embeddings)}/{len(self.chunks)}")
                    else:
                        # Add empty embedding to maintain index alignment
                        self.embeddings.append([0] * 1536)

                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    self.embeddings.append([0] * 1536)

        print(f"\n✅ Created {len(self.embeddings)} embeddings")

    def _create_single_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text chunk"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "text-embedding-ada-002",
            "input": text[:8000]  # Limit to first 8000 chars
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
                print(f"    Embedding API Error: {response.status_code}")
                return None

        except Exception as e:
            print(f"    Embedding Error: {e}")
            return None

    def save_vector_database(self):
        """Save the vector database to disk"""
        print(f"\n💾 Saving vector database to {self.db_path}...")

        db_data = {
            "chunks": self.chunks,
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "created_at": datetime.now().isoformat(),
            "total_chunks": len(self.chunks)
        }

        with open(self.db_path, 'wb') as f:
            pickle.dump(db_data, f)

        print(f"✅ Vector database saved!")
        print(f"   Database size: {os.path.getsize(self.db_path) / 1024 / 1024:.2f} MB")

    def load_vector_database(self):
        """Load the vector database from disk"""
        print(f"\n📂 Loading vector database from {self.db_path}...")

        try:
            with open(self.db_path, 'rb') as f:
                db_data = pickle.load(f)

            self.chunks = db_data["chunks"]
            self.embeddings = db_data["embeddings"]
            self.metadata = db_data["metadata"]

            print(f"✅ Loaded {len(self.chunks)} chunks from database")
            print(f"   Created on: {db_data.get('created_at', 'Unknown')}")
            return True

        except FileNotFoundError:
            print(f"❌ Vector database not found at {self.db_path}")
            return False
        except Exception as e:
            print(f"❌ Error loading database: {e}")
            return False

    def search_similar(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar chunks"""
        if not self.embeddings:
            print("❌ No embeddings in database. Please process the textbook first.")
            return []

        print(f"🔍 Searching for '{query}'...")

        # Create embedding for query
        query_embedding = self._create_single_embedding(query)
        if not query_embedding:
            print("❌ Failed to create query embedding")
            return []

        query_embedding = np.array(query_embedding)

        # Calculate similarities
        similarities = []
        for i, embedding in enumerate(self.embeddings):
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
                "chunk": self.chunks[i],
                "metadata": self.metadata[i],
                "similarity": similarity,
                "rank": i + 1
            })

        return results

    def ask_question(self, question: str, k: int = 3) -> Dict:
        """Ask a question with RAG"""
        # Search for relevant chunks
        relevant_chunks = self.search_similar(question, k)

        if not relevant_chunks:
            return {
                "question": question,
                "answer": "I couldn't find relevant information in the textbook.",
                "sources": []
            }

        # Create context
        context = ""
        for i, result in enumerate(relevant_chunks):
            context += f"\n\nSource {i+1} (Page {result['metadata']['page']}):\n{result['chunk'][:1000]}...\n"

        # Generate answer
        answer = self._generate_answer(question, context)

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "page": r["metadata"]["page"],
                    "similarity": r["similarity"],
                    "preview": r["chunk"][:200] + "..."
                }
                for r in relevant_chunks
            ]
        }

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using GPT"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": """You are a helpful assistant for the English textbook "English for Today" (Classes 9-10).
                    Answer questions based ONLY on the provided context. Be precise and helpful."""
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

    def get_stats(self):
        """Get database statistics"""
        return {
            "total_chunks": len(self.chunks),
            "total_characters": sum(len(c) for c in self.chunks),
            "average_chunk_size": np.mean([len(c) for c in self.chunks]) if self.chunks else 0,
            "pages_processed": len(set(m["page"] for m in self.metadata)),
            "database_size": os.path.getsize(self.db_path) / 1024 / 1024 if os.path.exists(self.db_path) else 0
        }


def main():
    """Main function to run the RAG system"""
    pdf_path = "Secondary - 2018 - Class - 9&10 - English for today-9-10  PDF Web .pdf"
    db_path = "textbook_vector_db.pkl"

    # Initialize RAG system
    rag = FullRAGSystem(pdf_path, db_path)

    # Check if database exists
    if os.path.exists(db_path):
        print("📚 Found existing vector database!")
        if rag.load_vector_database():
            stats = rag.get_stats()
            print("\n📊 Database Statistics:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.2f}")
                else:
                    print(f"   {key}: {value}")
        else:
            print("⚠️  Failed to load database. Will create new one.")
            # Process and create new database
            if rag.process_full_textbook(max_pages=30):
                rag.create_embeddings()
                rag.save_vector_database()
    else:
        print("🆕 No database found. Processing textbook...")
        print("⚠️  This will take a while and use API credits...")

        # Process the textbook
        if rag.process_full_textbook(max_pages=30):
            rag.create_embeddings()
            rag.save_vector_database()

            stats = rag.get_stats()
            print("\n📊 Database Statistics:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.2f}")
                else:
                    print(f"   {key}: {value}")

    # Interactive Q&A
    print("\n🤖 RAG System Ready!")
    print("Type 'quit' to exit or 'stats' to see database info")

    while True:
        question = input("\n📝 Ask a question: ").strip()

        if question.lower() == 'quit':
            break
        elif question.lower() == 'stats':
            stats = rag.get_stats()
            print("\n📊 Database Statistics:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.2f}")
                else:
                    print(f"   {key}: {value}")
            continue

        if question:
            print("\n🔍 Searching...")
            result = rag.ask_question(question, k=3)

            print(f"\n💡 Answer: {result['answer']}")

            if result['sources']:
                print(f"\n📚 Sources:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"   {i}. Page {source['page']} (similarity: {source['similarity']:.2f})")


if __name__ == "__main__":
    main()