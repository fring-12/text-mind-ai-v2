"""
Extract text from PDF using different methods
"""

import fitz  # PyMuPDF
import os

def extract_text_with_pymupdf(pdf_path):
    """Extract text using PyMuPDF with different settings"""
    print(f"📖 Opening PDF: {pdf_path}")

    # Try opening with different options
    try:
        # First attempt - normal
        print("\nTrying normal text extraction...")
        document = fitz.open(pdf_path)

        total_pages = len(document)
        print(f"Total pages: {total_pages}")

        # Check specific pages
        test_pages = [0, 1, 10, 50, 100, 150, 200]

        for page_num in test_pages:
            if page_num < total_pages:
                page = document[page_num]

                # Get text
                text = page.get_text()

                # Get images (in case it's image-based PDF)
                images = page.get_images()

                print(f"\nPage {page_num + 1}:")
                print(f"  Text length: {len(text)} characters")
                print(f"  Images found: {len(images)}")

                if len(text) > 100:
                    print(f"  Sample text: {text[:200]}...")
                else:
                    # Try OCR on this page
                    print("  Trying to get text as image...")
                    pix = page.get_pixmap()
                    print(f"  Image dimensions: {pix.width}x{pix.height}")

        # Check if it's an image-based PDF
        print("\n\nAnalyzing PDF type...")
        total_text = 0
        total_images = 0

        for page_num in range(min(10, total_pages)):
            page = document[page_num]
            text = page.get_text()
            images = page.get_images()
            total_text += len(text)
            total_images += len(images)

        print(f"First 10 pages:")
        print(f"  Total text: {total_text} characters")
        print(f"  Total images: {total_images}")

        if total_text < 1000:
            print("⚠️  This appears to be an image-based PDF")
            print("   You may need OCR to extract text")
        else:
            print("✅ Text-based PDF detected")

        document.close()

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    pdf_path = "Secondary - 2018 - Class - 9&10 - English for today-9-10  PDF Web .pdf"

    if os.path.exists(pdf_path):
        extract_text_with_pymupdf(pdf_path)
    else:
        print(f"❌ File not found: {pdf_path}")