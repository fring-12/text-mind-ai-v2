"""
PDF Text Processor Module
Handles extraction and preprocessing of text from PDF files
"""

import fitz  # PyMuPDF
from typing import List, Dict
import re


class PDFProcessor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract_text_from_pdf(self) -> Dict[str, str]:
        """
        Extract text from PDF with page numbers

        Returns:
            Dict with page numbers as keys and text as values
        """
        document = fitz.open(self.pdf_path)
        pages_text = {}

        for page_num in range(len(document)):
            page = document[page_num]
            text = page.get_text()

            # Clean up text
            text = self._clean_text(text)

            if text.strip():  # Only add non-empty pages
                pages_text[f"page_{page_num + 1}"] = text

        document.close()
        return pages_text

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove page numbers and headers/footers patterns
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'English For Today', '', text)

        # Fix common OCR issues
        text = text.replace('\n', ' ')

        return text.strip()

    def extract_structured_content(self) -> List[Dict]:
        """
        Extract structured content with unit/lesson information

        Returns:
            List of dictionaries with content metadata
        """
        pages_text = self.extract_text_from_pdf()
        structured_content = []

        current_unit = None
        current_lesson = None

        for page_num, text in pages_text.items():
            # Try to identify unit and lesson titles
            unit_match = re.search(r'Unit\s+\d+[:\s]+([^\n]+)', text, re.IGNORECASE)
            lesson_match = re.search(r'Lesson\s+\d+[:\s]+([^\n]+)', text, re.IGNORECASE)

            if unit_match:
                current_unit = unit_match.group(1).strip()

            if lesson_match:
                current_lesson = lesson_match.group(1).strip()

            structured_content.append({
                'page': page_num,
                'text': text,
                'unit': current_unit,
                'lesson': current_lesson
            })

        return structured_content


if __name__ == "__main__":
    # Test the processor
    processor = PDFProcessor("Secondary - 2018 - Class - 9&10 - English for today-9-10  PDF Web .pdf")

    # Extract structured content
    content = processor.extract_structured_content()

    print(f"Extracted {len(content)} pages")

    # Print first few pages as sample
    for i in range(min(3, len(content))):
        print(f"\n--- {content[i]['page']} ---")
        print(f"Unit: {content[i]['unit']}")
        print(f"Lesson: {content[i]['lesson']}")
        print(f"Text preview: {content[i]['text'][:200]}...")