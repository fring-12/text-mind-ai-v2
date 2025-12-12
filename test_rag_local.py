"""
Test RAG System Locally
"""

import os
import sys

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print(f"Current directory: {os.getcwd()}")
print(f"Database exists: {os.path.exists('textbook_vector_db.pkl')}")

if os.path.exists('textbook_vector_db.pkl'):
    # Import and test the RAG system
    from full_rag_system import FullRAGSystem

    # Initialize
    rag = FullRAGSystem("Secondary - 2018 - Class - 9&10 - English for today-9-10  PDF Web .pdf")

    # Load database
    if rag.load_vector_database():
        print("\n✅ Database loaded successfully!")

        # Test a query
        test_questions = [
            "What is Unit 1 about?",
            "Tell me about Good Citizens"
        ]

        for question in test_questions:
            print(f"\n📝 Q: {question}")
            result = rag.ask_question(question)
            print(f"💡 A: {result['answer'][:200]}...")
    else:
        print("\n❌ Failed to load database")