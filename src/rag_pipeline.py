"""
RAG Pipeline Module
Handles the complete Retrieval-Augmented Generation pipeline
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.embedding_manager import EmbeddingManager, VectorStoreManager

load_dotenv()


class RAGPipeline:
    def __init__(
        self,
        embedding_model: str = "openai",
        vector_db_type: str = "chroma",
        llm_provider: str = "openai",
        persist_directory: str = "./chroma_db"
    ):
        self.embedding_model = embedding_model
        self.vector_db_type = vector_db_type
        self.llm_provider = llm_provider
        self.persist_directory = persist_directory

        # Initialize components
        self.embedding_manager = EmbeddingManager(embedding_model)
        self.vector_store_manager = VectorStoreManager(vector_db_type, persist_directory)
        self.llm = self._initialize_llm()

        # Load existing vector store
        try:
            self.vector_store_manager.load_vector_store(self.embedding_manager.embeddings)
        except Exception as e:
            print(f"Warning: Could not load vector store: {e}")
            print("Please run the embedding creation script first")

        # Initialize QA chain
        self.qa_chain = self._create_qa_chain()

    def _initialize_llm(self):
        """Initialize the language model"""
        if self.llm_provider.lower() == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            return ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        elif self.llm_provider.lower() == "huggingface":
            if not os.getenv("HUGGINGFACE_API_KEY"):
                raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")
            return HuggingFaceHub(
                repo_id="google/flan-t5-large",
                huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _create_qa_chain(self):
        """Create the question-answering chain"""
        # Define prompt template
        template = """
        Use the following context from the English textbook "English for Today" for classes 9-10 to answer the question.
        If you don't know the answer from the context, just say that you don't know. Don't try to make up an answer.

        Context:
        {context}

        Question: {question}

        Helpful Answer:"""

        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # Create chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store_manager.vector_store.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

        return qa_chain

    def query(self, question: str, k: int = 4) -> Dict:
        """
        Query the RAG system

        Args:
            question: User's question
            k: Number of documents to retrieve

        Returns:
            Dictionary with answer and source documents
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized")

        # Update retriever's k parameter
        self.qa_chain.retriever.search_kwargs['k'] = k

        # Query the chain
        result = self.qa_chain({"query": question})

        # Format sources
        sources = []
        for doc in result['source_documents']:
            sources.append({
                'content': doc.page_content,
                'metadata': doc.metadata
            })

        return {
            'question': question,
            'answer': result['result'],
            'sources': sources
        }

    def get_similar_documents(self, query: str, k: int = 4) -> List[Document]:
        """
        Get similar documents without generation

        Args:
            query: Query string
            k: Number of documents to retrieve

        Returns:
            List of similar documents
        """
        return self.vector_store_manager.similarity_search(query, k=k)

    def query_with_context(self, question: str, context: Optional[str] = None) -> Dict:
        """
        Query with additional context

        Args:
            question: User's question
            context: Additional context to include

        Returns:
            Dictionary with answer and source documents
        """
        if context:
            # Augment the question with context
            augmented_question = f"Additional Context: {context}\n\nQuestion: {question}"
            return self.query(augmented_question)
        else:
            return self.query(question)


# Example usage
if __name__ == "__main__":
    # Initialize RAG pipeline
    rag = RAGPipeline()

    # Example queries
    test_queries = [
        "What is the importance of learning English?",
        "Explain the concept of narrative writing",
        "What are the different parts of speech?",
        "How do we write a formal letter?"
    ]

    print("\n🤖 RAG System Test Queries\n")

    for query in test_queries:
        print(f"\n📝 Question: {query}")
        result = rag.query(query, k=3)

        print(f"\n💡 Answer: {result['answer']}")
        print(f"\n📚 Sources: {len(result['sources'])} documents retrieved")

        for i, source in enumerate(result['sources'][:2]):
            print(f"\n   Source {i+1}:")
            print(f"   - Page: {source['metadata'].get('page', 'Unknown')}")
            print(f"   - Unit: {source['metadata'].get('unit', 'Unknown')}")
            print(f"   - Preview: {source['content'][:150]}...")

        print("-" * 80)