import json
import os
import numpy as np
import faiss
from typing import Dict, List, Tuple

class FAISSVectorStore:
    def __init__(self, dimension: int = 384):  # default dimension for all-MiniLM-L6-v2 embeddings
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []  # Store original texts
        self.metadata = []  # Store metadata like source file, chunk number

    def add_embeddings(self, embeddings_file: str):
        """Add embeddings from a JSON file to the FAISS index"""
        try:
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract embeddings and texts
            embeddings = []
            for chunk_id, chunk_data in data.items():
                embeddings.append(chunk_data['embedding'])
                self.texts.append(chunk_data['text'])
                self.metadata.append({
                    'source': os.path.basename(embeddings_file),
                    'chunk_id': chunk_id
                })
            
            # Convert to numpy array and add to FAISS index
            embeddings_array = np.array(embeddings).astype('float32')
            self.index.add(embeddings_array)
            
            print(f"Added {len(embeddings)} embeddings from {os.path.basename(embeddings_file)}")
            
        except Exception as e:
            print(f"Error processing {embeddings_file}: {str(e)}")

    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[str, float, Dict]]:
        """Search for similar texts using a query embedding"""
        query_array = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_array, k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.texts):  # Ensure index is valid
                results.append((self.texts[idx], dist, self.metadata[idx]))
        
        return results

    def save_index(self, save_dir: str):
        """Save the FAISS index and associated data"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(save_dir, 'faiss_index.bin')
        faiss.write_index(self.index, index_path)
        
        # Save texts and metadata
        data_path = os.path.join(save_dir, 'vector_store_data.json')
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump({
                'texts': self.texts,
                'metadata': self.metadata
            }, f, ensure_ascii=False, indent=2)
        
        print(f"Saved vector store to {save_dir}")

    @classmethod
    def load(cls, save_dir: str) -> 'FAISSVectorStore':
        """Load a saved FAISS index and associated data"""
        vector_store = cls()
        
        # Load FAISS index
        index_path = os.path.join(save_dir, 'faiss_index.bin')
        vector_store.index = faiss.read_index(index_path)
        
        # Load texts and metadata
        data_path = os.path.join(save_dir, 'vector_store_data.json')
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            vector_store.texts = data['texts']
            vector_store.metadata = data['metadata']
        
        return vector_store

def create_vector_store(data_dir: str, save_dir: str = 'vector_store'):
    """Create a vector store from all embedding files in the data directory"""
    vector_store = FAISSVectorStore()
    
    for filename in os.listdir(data_dir):
        if filename.endswith('_embeddings.json'):
            embeddings_file = os.path.join(data_dir, filename)
            vector_store.add_embeddings(embeddings_file)
    
    # Save the vector store
    vector_store.save_index(save_dir)
    return vector_store

if __name__ == "__main__":
    data_dir = "data"
    save_dir = "vector_store"
    
    # Create and save vector store
    vector_store = create_vector_store(data_dir, save_dir)
    print(f"\nVector store creation complete! Total texts indexed: {len(vector_store.texts)}")