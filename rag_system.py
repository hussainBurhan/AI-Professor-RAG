from transformers import AutoTokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import faiss
import json
import os
import numpy as np
import torch

class RAGSystem:
    def __init__(self, vector_store_dir: str):
        """Initialize the RAG system with free models"""
        # Load the embedding model (already using free SentenceTransformer)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load free LLM from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
        
        # Load the vector store
        self.index = faiss.read_index(os.path.join(vector_store_dir, 'faiss_index.bin'))
        with open(os.path.join(vector_store_dir, 'vector_store_data.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.texts = data['texts']
            self.metadata = data['metadata']

    def get_relevant_context(self, query: str, k: int = 3) -> list:
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
        """Generate a response using RAG with free models"""
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
        
        # Generate response using T5
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=512,
            min_length=50,
            do_sample=True,  # Enable sampling
            temperature=0.7,  # Control randomness
            num_beams=4,
            no_repeat_ngram_size=2,
            top_k=50,        # Limit vocabulary choices
            top_p=0.95       # Nucleus sampling
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

def main():
    # Initialize the RAG system
    vector_store_dir = "vector_store"
    rag = RAGSystem(vector_store_dir)
    
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