"""
Main script to run the RAG system pipeline
"""

import os
from dotenv import load_dotenv
from src.pdf_processor import PDFProcessor
from src.text_chunker import TextChunker

# Load environment variables
load_dotenv()

def main():
    # Configuration
    PDF_PATH = "Secondary - 2018 - Class - 9&10 - English for today-9-10  PDF Web .pdf"
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))

    print("🚀 Starting RAG System Setup...")
    print(f"📄 Processing PDF: {PDF_PATH}")

    # Step 1: Extract text from PDF
    print("\n📖 Step 1: Extracting text from PDF...")
    processor = PDFProcessor(PDF_PATH)
    structured_content = processor.extract_structured_content()
    print(f"✅ Extracted content from {len(structured_content)} pages")

    # Step 2: Chunk the text
    print("\n✂️ Step 2: Chunking text...")
    chunker = TextChunker(CHUNK_SIZE, CHUNK_OVERLAP)
    chunks = chunker.chunk_text(structured_content)
    print(f"✅ Created {len(chunks)} chunks")

    # Display statistics
    print("\n📊 Statistics:")
    print(f"- Total pages processed: {len(structured_content)}")
    print(f"- Total chunks created: {len(chunks)}")
    print(f"- Average chunk size: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} characters")

    # Save chunks to file for inspection
    with open("chunks_sample.txt", "w", encoding="utf-8") as f:
        f.write("Sample of chunks created:\n\n")
        for i, chunk in enumerate(chunks[:5]):  # Save first 5 chunks as sample
            f.write(f"--- Chunk {i+1} ---\n")
            f.write(f"Metadata: {chunk.metadata}\n")
            f.write(f"Content: {chunk.page_content[:500]}...\n\n")

    print("\n✅ Chunk sample saved to 'chunks_sample.txt'")
    print("\n🎯 Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Set up your .env file with API keys")
    print("3. Run embedding creation script")
    print("4. Set up vector database")
    print("5. Create query interface")


if __name__ == "__main__":
    main()