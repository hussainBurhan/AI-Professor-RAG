# AI Professor RAG System

An intelligent question-answering system powered by Retrieval-Augmented Generation (RAG) that specializes in answering questions about Physics, Biology, and Political Science using educational textbook content.

## Overview

The AI Professor RAG System combines the power of modern language models with a sophisticated retrieval system to provide accurate, context-aware answers to academic questions. It uses:
- FAISS for efficient vector similarity search
- SentenceTransformer (all-MiniLM-L6-v2) for text embeddings
- Google's FLAN-T5 Base model for text generation
- RAG architecture for context-enhanced responses

## Project Structure

AI Professor RAG/
├── rag_system.py        # Main RAG implementation
├── pdf_processor.py     # PDF text extraction
├── text_chunker.py      # Text chunking
├── embeddings_creator.py # Embeddings generation
├── vector_store.py      # FAISS vector store management
├── data/               # Directory for source PDFs and processed files
└── vector_store/        # Directory containing indexed knowledge

## Prerequisites

- Python 3.7+
- PyTorch
- Transformers
- SentenceTransformers
- FAISS
- NumPy
- PyPDF2

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Install the required packages:
```bash
pip install transformers sentence-transformers faiss-cpu torch numpy PyPDF2 langchain-text-splitters
```

## Data Preparation and Processing

1. Download the dataset:
   - Get the ebook PDFs from [Kaggle Dataset](https://www.kaggle.com/datasets/rohanthoma/ebook-pdfs)
   - Create a `data` folder in the project root
   - Save the downloaded books in the `data` folder

2. Process the data (run these scripts in order):
   ```bash
   python pdf_processor.py      # Extract text from PDFs
   python text_chunker.py       # Split texts into chunks
   python embeddings_creator.py # Create embeddings
   python vector_store.py      # Build FAISS vector store
   ```

## Usage

1. After completing the data processing steps above, run the system:
```bash
python rag_system.py
```

2. Start asking questions! The system will:
   - Take your input question
   - Retrieve relevant context from the knowledge base
   - Generate a comprehensive answer using the retrieved context

## Workflow

1. **Data Processing Pipeline**:
   - PDF Processing: Extract text from PDF books
   - Text Chunking: Split large texts into manageable chunks
   - Embeddings Creation: Generate vector embeddings for each chunk
   - Vector Store Creation: Index embeddings in FAISS for efficient retrieval

2. **Query Processing**:
   - User inputs a question
   - Question is embedded using SentenceTransformer

3. **Context Retrieval**:
   - System searches for relevant context using FAISS
   - Top-k most similar text passages are retrieved
   - Source information is preserved

4. **Response Generation**:
   - Retrieved contexts are formatted into a prompt
   - FLAN-T5 generates a comprehensive answer
   - Response is presented to the user

## Features

- Free and open-source models
- Efficient vector similarity search
- Source attribution for answers
- Support for multiple academic subjects
- Interactive command-line interface

## Limitations

- Knowledge is limited to pre-indexed content
- Response quality depends on available context
- Processing time may vary based on hardware
- Initial setup requires downloading and processing PDFs

## Exit

Type 'quit' to exit the system.
```

        