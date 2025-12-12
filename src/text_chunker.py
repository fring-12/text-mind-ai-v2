"""
Text Chunking Module
Splits extracted text into manageable chunks for embedding
"""

from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_text(self, structured_content: List[Dict]) -> List[Document]:
        """
        Chunk structured content into smaller pieces

        Args:
            structured_content: List of dictionaries with page content

        Returns:
            List of Document objects with chunked text
        """
        documents = []
        chunk_id = 0

        for content in structured_content:
            # Create Document from page content
            doc = Document(
                page_content=content['text'],
                metadata={
                    'page': content['page'],
                    'unit': content['unit'],
                    'lesson': content['lesson']
                }
            )

            # Split the document
            chunks = self.text_splitter.split_documents([doc])

            # Add chunk IDs and update metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = chunk_id
                chunk.metadata['chunk_index'] = i
                chunk.metadata['total_chunks'] = len(chunks)
                documents.append(chunk)
                chunk_id += 1

        return documents

    def create_semantic_chunks(self, structured_content: List[Dict]) -> List[Document]:
        """
        Create chunks based on semantic boundaries (lessons/units)

        Args:
            structured_content: List of dictionaries with page content

        Returns:
            List of Document objects with semantically grouped chunks
        """
        documents = []
        chunk_id = 0

        # Group by unit and lesson
        unit_lesson_groups = {}
        for content in structured_content:
            key = (content['unit'], content['lesson'])
            if key not in unit_lesson_groups:
                unit_lesson_groups[key] = []
            unit_lesson_groups[key].append(content)

        # Create chunks for each group
        for (unit, lesson), pages in unit_lesson_groups.items():
            combined_text = "\n\n".join([page['text'] for page in pages])

            doc = Document(
                page_content=combined_text,
                metadata={
                    'unit': unit,
                    'lesson': lesson,
                    'pages': [page['page'] for page in pages],
                    'content_type': 'lesson_chunk'
                }
            )

            # If lesson is too long, split it
            if len(combined_text) > self.chunk_size:
                chunks = self.text_splitter.split_documents([doc])
                for i, chunk in enumerate(chunks):
                    chunk.metadata['chunk_id'] = chunk_id
                    chunk.metadata['chunk_index'] = i
                    chunk.metadata['total_chunks'] = len(chunks)
                    documents.append(chunk)
                    chunk_id += 1
            else:
                doc.metadata['chunk_id'] = chunk_id
                doc.metadata['chunk_index'] = 0
                doc.metadata['total_chunks'] = 1
                documents.append(doc)
                chunk_id += 1

        return documents


if __name__ == "__main__":
    # Test the chunker
    from pdf_processor import PDFProcessor

    # Extract content
    processor = PDFProcessor("Secondary - 2018 - Class - 9&10 - English for today-9-10  PDF Web .pdf")
    content = processor.extract_structured_content()

    # Create chunks
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_text(content)

    print(f"Created {len(chunks)} chunks")

    # Print first chunk as example
    if chunks:
        print("\n--- First Chunk ---")
        print(f"Content: {chunks[0].page_content[:300]}...")
        print(f"Metadata: {chunks[0].metadata}")