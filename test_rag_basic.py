"""
Basic Test for PDF Processing and RAG
"""

import fitz  # PyMuPDF
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Please set OPENAI_API_KEY in your .env file")
    exit(1)

print("✅ OpenAI API key found")

# Test PDF processing
pdf_path = "Secondary - 2018 - Class - 9&10 - English for today-9-10  PDF Web .pdf"

if not os.path.exists(pdf_path):
    print(f"❌ PDF file not found: {pdf_path}")
    exit(1)

print(f"✅ PDF file found: {pdf_path}")

# Extract text from PDF
print("\n📖 Extracting text from PDF...")
document = fitz.open(pdf_path)
total_pages = len(document)
print(f"   Total pages: {total_pages}")

# Extract first few pages as test
sample_text = ""
for page_num in range(min(3, total_pages)):
    page = document[page_num]
    text = page.get_text()
    if text.strip():
        sample_text += f"\n\n--- Page {page_num + 1} ---\n\n{text[:500]}..."

document.close()

print("\n📝 Sample extracted text:")
print(sample_text[:1000])
print("...")

# Simple chunking test
print(f"\n✂️ Creating text chunks...")
chunk_size = 500
chunks = []
for i in range(0, len(sample_text), chunk_size):
    chunks.append(sample_text[i:i + chunk_size])

print(f"   Created {len(chunks)} test chunks")

print("\n✅ PDF processing test completed successfully!")
print("\n📊 Statistics:")
print(f"   - Total pages in PDF: {total_pages}")
print(f"   - Sample text length: {len(sample_text)} characters")
print(f"   - Number of test chunks: {len(chunks)}")

print("\n🎯 Next steps:")
print("1. The PDF processing works correctly")
print("2. You can now create embeddings with OpenAI")
print("3. Then build the full RAG system")