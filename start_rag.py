#!/usr/bin/env python3
"""
Start the RAG System
"""

import os
import sys
import subprocess
import time

def check_database():
    """Check if vector database exists"""
    if os.path.exists("textbook_vector_db.pkl"):
        size = os.path.getsize("textbook_vector_db.pkl") / 1024 / 1024
        print(f"✅ Vector database found ({size:.2f} MB)")
        return True
    else:
        print("❌ Vector database not found!")
        print("   Run 'python3 full_rag_system.py' first to create it.")
        return False

def start_web_app():
    """Start the Streamlit web app"""
    print("\n🚀 Starting RAG web app...")
    print("   Open http://localhost:8501 in your browser")
    print("\n   Or access remotely:")
    print("   - Network URL will be shown below")

    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "rag_web_app.py",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n👋 Stopping RAG system...")

def main():
    print("=" * 50)
    print("📚 English Textbook RAG System")
    print("=" * 50)

    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Check requirements
    if not check_database():
        return

    # Start web app
    start_web_app()

if __name__ == "__main__":
    main()