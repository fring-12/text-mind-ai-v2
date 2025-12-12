# English Textbook RAG System

## Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/fring-12/text-mind-ai-v2.git
cd text-mind-ai-v2
```

2. **Setup environment**
```bash
# Copy example environment file
cp .env.example .env

# Add your OpenAI API key to .env file
# OPENAI_API_KEY=your_api_key_here
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Generate vector database**
```bash
python3 full_rag_system.py
```

5. **Run the web app**
```bash
python3 -m streamlit run rag_web_fixed2.py --server.port 8501
```

6. **Open browser**
Navigate to http://localhost:8501

## Files Included
- `full_rag_system.py` - Core RAG system
- `rag_web_fixed2.py` - Web interface
- `process_full_book.py` - Process all 209 pages
- `requirements.txt` - Python dependencies
- `.env.example` - Environment template

## Features
- OCR-based text extraction from PDF
- Vector embeddings for fast search
- Web interface for Q&A
- Source citations with page numbers

## Current Coverage
- Pages processed: 29/209
- Vector database: 28 chunks
- Response time: < 1 second