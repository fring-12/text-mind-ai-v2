"""
Simple RAG System Demo
A minimal implementation without complex dependencies
"""

import fitz  # PyMuPDF
import os
import json
from typing import List, Dict
import openai
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SimpleRAG:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.chunks = []
        self.embeddings = []

        # Initialize OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Please set OPENAI_API_KEY in your .env file")
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Process PDF
        self.process_pdf()

    def process_pdf(self):
        """Extract and chunk text from PDF"""
        print("📖 Processing PDF...")
        document = fitz.open(self.pdf_path)

        all_text = ""
        for page_num in range(len(document)):
            page = document[page_num]
            text = page.get_text()
            if text.strip():
                all_text += f"\n\n--- Page {page_num + 1} ---\n\n{text}"

        document.close()

        # Simple chunking
        chunk_size = 1000
        for i in range(0, len(all_text), chunk_size):
            chunk = all_text[i:i + chunk_size]
            self.chunks.append({
                "id": i,
                "text": chunk,
                "page_start": (i // chunk_size) + 1
            })

        print(f"✅ Created {len(self.chunks)} text chunks")

    def create_embeddings(self):
        """Create embeddings for all chunks"""
        print("🧠 Creating embeddings...")
        batch_size = 10

        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            texts = [chunk["text"] for chunk in batch]

            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model="text-embedding-ada-002"
                )

                for j, embedding in enumerate(response.data):
                    self.embeddings.append(embedding.embedding)

                print(f"  Processed {min(i + batch_size, len(self.chunks))}/{len(self.chunks)} chunks")

            except Exception as e:
                print(f"Error creating embeddings: {e}")
                break

        print(f"✅ Created {len(self.embeddings)} embeddings")

    def similarity_search(self, query: str, k: int = 3) -> List[Dict]:
        """Find most similar chunks"""
        # Create embedding for query
        response = self.client.embeddings.create(
            input=[query],
            model="text-embedding-ada-002"
        )
        query_embedding = np.array(response.data[0].embedding)

        # Calculate similarities
        similarities = []
        for i, chunk_embedding in enumerate(self.embeddings):
            chunk_embedding = np.array(chunk_embedding)
            similarity = np.dot(query_embedding, chunk_embedding)
            similarities.append((i, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top k chunks
        results = []
        for i, similarity in similarities[:k]:
            results.append({
                "chunk": self.chunks[i],
                "similarity": similarity
            })

        return results

    def ask_question(self, question: str) -> str:
        """Ask a question and get an answer"""
        # Find relevant chunks
        relevant_chunks = self.similarity_search(question, k=3)

        # Create context
        context = ""
        for i, result in enumerate(relevant_chunks):
            context += f"\nContext {i+1} (Page {result['chunk']['page_start']}):\n{result['chunk']['text'][:500]}...\n"

        # Generate answer
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant answering questions about an English textbook. Use only the provided context to answer the question."
                },
                {
                    "role": "user",
                    "content": f"Context from the textbook:{context}\n\nQuestion: {question}\n\nAnswer:"
                }
            ],
            max_tokens=500,
            temperature=0
        )

        return response.choices[0].message.content

    def save_embeddings(self, filename: str = "embeddings.json"):
        """Save embeddings to file"""
        data = {
            "chunks": self.chunks,
            "embeddings": self.embeddings
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"✅ Saved embeddings to {filename}")

    def load_embeddings(self, filename: str = "embeddings.json"):
        """Load embeddings from file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        self.chunks = data["chunks"]
        self.embeddings = data["embeddings"]
        print(f"✅ Loaded {len(self.chunks)} chunks from {filename}")


def main():
    # Check if embeddings already exist
    if os.path.exists("embeddings.json"):
        print("Found existing embeddings, loading...")
        rag = SimpleRAG("Secondary - 2018 - Class - 9&10 - English for today-9-10  PDF Web .pdf")
        rag.load_embeddings()
    else:
        print("No embeddings found, creating new ones...")
        rag = SimpleRAG("Secondary - 2018 - Class - 9&10 - English for today-9-10  PDF Web .pdf")
        rag.create_embeddings()
        rag.save_embeddings()

    # Interactive question answering
    print("\n🤖 RAG System Ready!")
    print("Type 'quit' to exit\n")

    while True:
        question = input("\n📝 Ask a question: ")
        if question.lower() == 'quit':
            break

        try:
            print("\n🔍 Searching...")
            answer = rag.ask_question(question)
            print(f"\n💡 Answer: {answer}")
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()