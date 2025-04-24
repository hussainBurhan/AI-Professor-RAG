from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import faiss
import json
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class RAGSystem:
    def __init__(self, vector_store_dir: str, gemini_api_key: str):
        """Initialize the RAG system with Gemini"""
        # Load the embedding model (already using free SentenceTransformer)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Load the vector store
        self.index = faiss.read_index(os.path.join(vector_store_dir, 'faiss_index.bin'))
        with open(os.path.join(vector_store_dir, 'vector_store_data.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.texts = data['texts']
            self.metadata = data['metadata']

    def get_relevant_context(self, query: str, k: int = 10) -> list:
        """Retrieve relevant context for the query"""
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        query_embedding = query_embedding.astype('float32')
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        # Get relevant texts and their sources
        contexts = []
        for idx in indices[0]:
            if idx < len(self.texts):
                contexts.append({
                    'text': self.texts[idx],
                    'source': self.metadata[idx]['source']
                })
        
        return contexts

    def generate_response(self, query: str) -> str:
        """Generate a response using RAG with Gemini"""
        # Get relevant context
        contexts = self.get_relevant_context(query)
        
        # Prepare prompt with context
        context_text = "\n\n".join([f"From {ctx['source']}:\n{ctx['text']}" for ctx in contexts])
        prompt = f"""Based on the following context from educational textbooks, please provide a clear and comprehensive answer to the question.
        If the context doesn't contain relevant information, please say so.
        Use the context to provide specific examples and explanations.

        Context:
        {context_text}

        Question: {query}

        Answer:"""
        
        # Generate response using Gemini
        response = self.model.generate_content(prompt)
        return response.text

def main():
    # Initialize the RAG system
    vector_store_dir = "vector_store"
    gemini_api_key = os.getenv("GEMINI_API_KEY")  # Get API key from environment variable
    if not gemini_api_key:
        raise ValueError("Please set the GEMINI_API_KEY environment variable")
        
    rag = RAGSystem(vector_store_dir, gemini_api_key)
    
    print("AI Professor RAG System")
    print("Ask me anything about Physics, Biology, or Political Science!")
    print("Type 'quit' to exit")
    
    while True:
        query = input("\nYour question: ")
        if query.lower() == 'quit':
            break
            
        try:
            response = rag.generate_response(query)
            print("\nAI Professor's Answer:")
            print(response)
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()