# 📚 English Textbook RAG System

A Retrieval-Augmented Generation (RAG) system built to answer questions about the "English for Today" textbook for classes 9-10.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```env
# For OpenAI (recommended)
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL=openai
LLM_PROVIDER=openai

# Or use HuggingFace alternatives
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
EMBEDDING_MODEL=huggingface
LLM_PROVIDER=huggingface

# Vector database choice
VECTOR_DB=chroma
```

### 3. Process the PDF and Create Embeddings

Run the main script to:
- Extract text from the PDF
- Chunk the text into manageable segments
- Create embeddings
- Store in vector database

```bash
python main.py
```

### 4. Launch the Web Interface

```bash
streamlit run app.py
```

The web interface will open at `http://localhost:8501`

## 📁 Project Structure

```
text-mind-ai/
├── src/
│   ├── pdf_processor.py    # PDF text extraction
│   ├── text_chunker.py     # Text chunking logic
│   ├── embedding_manager.py # Embedding and vector store management
│   └── rag_pipeline.py     # Complete RAG pipeline
├── main.py                 # Main script for setup
├── app.py                  # Streamlit web interface
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
└── README.md              # This file
```

## 🛠️ How It Works

1. **PDF Processing**: The system extracts text from the English textbook PDF
2. **Text Chunking**: Text is split into smaller, manageable chunks with overlap
3. **Embedding Creation**: Each chunk is converted into a vector embedding
4. **Vector Storage**: Embeddings are stored in a vector database (Chroma/FAISS)
5. **Query Processing**: When you ask a question:
   - The question is converted to an embedding
   - Similar text chunks are retrieved
   - An LLM generates an answer based on retrieved context

## 💡 Usage Examples

Try these example questions in the web interface:

- "What is the importance of learning English?"
- "Explain the concept of narrative writing"
- "What are the different parts of speech?"
- "How do we write a formal letter?"
- "What is a simile?"

## 🔧 Configuration Options

### Embedding Models
- **OpenAI**: `text-embedding-ada-002` (recommended)
- **HuggingFace**: `sentence-transformers/all-MiniLM-L6-v2`

### Vector Databases
- **Chroma**: Persistent storage on disk
- **FAISS**: In-memory or disk-based storage

### Language Models
- **OpenAI**: GPT-3.5-turbo or GPT-4
- **HuggingFace**: Various open models via HuggingFace Hub

## 📊 Performance Tips

1. **Chunk Size**: Adjust `CHUNK_SIZE` (default: 1000) for better context
2. **Overlap**: Use `CHUNK_OVERLAP` (default: 200) to maintain context between chunks
3. **Number of Sources**: Adjust k in the web interface for more/less sources

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure your API keys are correctly set in `.env`
   - Check that the keys have sufficient credits

2. **Memory Issues**
   - Reduce chunk size if processing large documents
   - Use FAISS instead of Chroma for large datasets

3. **Poor Answers**
   - Increase the number of retrieved sources
   - Adjust chunk size and overlap
   - Try a more powerful LLM (GPT-4 instead of GPT-3.5)

### Reset the System

To completely reset the vector database:
```bash
rm -rf chroma_db/
rm -rf faiss_index/
```

Then re-run `python main.py` to rebuild.

## 🤝 Contributing

Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests

## 📝 License

This project is for educational purposes. Please ensure you have proper rights to use the textbook content.

## 🔮 Future Enhancements

- [ ] Support for multiple PDFs
- [ ] Advanced chunking strategies
- [ ] Citations and references
- [ ] Export functionality
- [ ] User feedback system
- [ ] Multi-language support