"""
Embedding Manager Module
Handles creation and management of text embeddings
"""

import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS
import numpy as np

load_dotenv()


class EmbeddingManager:
    def __init__(self, embedding_model: str = "openai"):
        self.embedding_model = embedding_model
        self.embeddings = self._initialize_embeddings()

    def _initialize_embeddings(self):
        """Initialize the embedding model based on configuration"""
        if self.embedding_model.lower() == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            return OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        elif self.embedding_model.lower() == "huggingface":
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model}")

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of texts

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        print(f"Creating embeddings for {len(texts)} texts...")
        embeddings = self.embeddings.embed_documents(texts)
        print(f"✅ Created {len(embeddings)} embeddings")
        return embeddings

    def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for a single text

        Args:
            text: Single text string to embed

        Returns:
            Single embedding vector
        """
        return self.embeddings.embed_query(text)


class VectorStoreManager:
    def __init__(self, vector_db_type: str = "chroma", persist_directory: str = "./chroma_db"):
        self.vector_db_type = vector_db_type
        self.persist_directory = persist_directory
        self.vector_store = None

    def create_vector_store(self, documents: List[Document], embeddings):
        """
        Create and populate vector store

        Args:
            documents: List of Document objects
            embeddings: Embedding model instance
        """
        print(f"Creating {self.vector_db_type} vector store...")

        if self.vector_db_type.lower() == "chroma":
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=self.persist_directory
            )
            print(f"✅ Chroma DB created at {self.persist_directory}")

        elif self.vector_db_type.lower() == "faiss":
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=embeddings
            )
            # Save FAISS index
            self.vector_store.save_local("faiss_index")
            print("✅ FAISS index created and saved")

        else:
            raise ValueError(f"Unsupported vector database: {self.vector_db_type}")

    def load_vector_store(self, embeddings):
        """
        Load existing vector store

        Args:
            embeddings: Embedding model instance
        """
        if self.vector_db_type.lower() == "chroma":
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings
            )
            print(f"✅ Loaded Chroma DB from {self.persist_directory}")

        elif self.vector_db_type.lower() == "faiss":
            self.vector_store = FAISS.load_local(
                "faiss_index",
                embeddings=embeddings
            )
            print("✅ Loaded FAISS index")

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search

        Args:
            query: Query string
            k: Number of documents to retrieve

        Returns:
            List of similar documents
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")

        return self.vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """
        Perform similarity search with scores

        Args:
            query: Query string
            k: Number of documents to retrieve

        Returns:
            List of (Document, score) tuples
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")

        return self.vector_store.similarity_search_with_score(query, k=k)


if __name__ == "__main__":
    # Test the embedding and vector store creation
    from src.pdf_processor import PDFProcessor
    from src.text_chunker import TextChunker

    # Configuration
    PDF_PATH = "Secondary - 2018 - Class - 9&10 - English for today-9-10  PDF Web .pdf"
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai")
    VECTOR_DB = os.getenv("VECTOR_DB", "chroma")

    # Process PDF and create chunks
    print("Processing PDF...")
    processor = PDFProcessor(PDF_PATH)
    content = processor.extract_structured_content()
    chunker = TextChunker()
    chunks = chunker.chunk_text(content)

    # Create embeddings and vector store
    print("\nCreating embeddings...")
    embedding_manager = EmbeddingManager(EMBEDDING_MODEL)
    vector_store_manager = VectorStoreManager(VECTOR_DB)

    # Create vector store
    vector_store_manager.create_vector_store(chunks, embedding_manager.embeddings)

    # Test search
    print("\nTesting similarity search...")
    results = vector_store_manager.similarity_search("What is grammar?", k=2)
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {doc.metadata}")
        print(f"Content: {doc.page_content[:200]}...")