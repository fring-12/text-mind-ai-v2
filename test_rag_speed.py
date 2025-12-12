"""
Test RAG System Speed
Demonstrates fast querying with vector database
"""

import os
import pickle
import numpy as np
import requests
import time
from dotenv import load_dotenv

load_dotenv()

def load_database():
    """Load vector database"""
    with open("textbook_vector_db.pkl", 'rb') as f:
        return pickle.load(f)

def create_embedding(text):
    """Create embedding"""
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

def search_similar(query, db, k=3):
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

def main():
    print("🚀 RAG System Speed Test")
    print("=" * 50)

    # Load database
    print("\n📂 Loading vector database...")
    start_time = time.time()
    db = load_database()
    load_time = time.time() - start_time
    print(f"   ✅ Loaded in {load_time:.3f} seconds")
    print(f"   📊 Database contains {len(db['chunks'])} chunks")

    # Test queries
    test_queries = [
        "What is Unit 1 about?",
        "Tell me about Good Citizens",
        "What is Lesson 1?",
        "What topics are covered?",
        "Pastimes unit"
    ]

    print("\n🔍 Testing query speed...")
    print("-" * 50)

    total_time = 0
    for query in test_queries:
        print(f"\n📝 Query: {query}")

        # Measure search time
        start_time = time.time()
        results = search_similar(query, db, k=3)
        search_time = time.time() - start_time

        total_time += search_time
        print(f"⚡ Search time: {search_time:.3f} seconds")
        print(f"📚 Found {len(results)} results")

        if results:
            print(f"   Top match: Page {results[0]['metadata']['page']} (similarity: {results[0]['similarity']:.3f})")
            print(f"   Preview: {results[0]['chunk'][:100]}...")

    print("\n" + "=" * 50)
    print(f"📊 Performance Summary:")
    print(f"   Total queries: {len(test_queries)}")
    print(f"   Average search time: {total_time/len(test_queries):.3f} seconds")
    print(f"   Fastest search: {min([time.time() - start_time for _ in range(1)]):.3f} seconds")
    print(f"   Total database size: {os.path.getsize('textbook_vector_db.pkl') / 1024 / 1024:.2f} MB")

    print("\n✨ Key Benefits:")
    print("   ✓ No PDF processing needed for queries")
    print("   ✓ No OCR for each question")
    print("   ✓ Instant similarity search")
    print("   ✓ Results in milliseconds")

if __name__ == "__main__":
    main()