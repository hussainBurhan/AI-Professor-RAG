from sentence_transformers import SentenceTransformer
import os
import json
from typing import Dict, List

class EmbeddingsCreator:
    def __init__(self):
        """Initialize the embeddings creator with Sentence Transformers"""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def create_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Create embeddings for a list of text chunks"""
        embeddings = self.model.encode(chunks)
        return embeddings.tolist()  # Convert numpy array to list

    def process_chunk_file(self, file_path: str) -> Dict[str, dict]:
        """Process a chunks file and create embeddings for each chunk"""
        chunk_embeddings = {}
        current_chunk = ""
        chunk_number = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                chunks = []
                current_chunk = ""
                
                # Split content by the separator used in chunk files
                for line in content.split('\n'):
                    if line.startswith('Chunk '):
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = ""
                    elif line.startswith('='*50):
                        continue
                    else:
                        current_chunk += line + "\n"
                
                # Add the last chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Create embeddings for all chunks
                embeddings = self.create_embeddings(chunks)
                
                # Create a dictionary with chunk text and its embedding
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    chunk_embeddings[f"chunk_{i+1}"] = {
                        "text": chunk,
                        "embedding": embedding
                    }
                
                # Save embeddings to a JSON file
                output_file = file_path.replace('_chunks.txt', '_embeddings.json')
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk_embeddings, f, ensure_ascii=False, indent=2)
                
                print(f"Created embeddings for {len(chunks)} chunks from {os.path.basename(file_path)}")
                print(f"Saved embeddings to {os.path.basename(output_file)}")
                
                return chunk_embeddings
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return {}

    def process_all_chunk_files(self, data_dir: str):
        """Process all chunk files in the data directory"""
        all_embeddings = {}
        
        for filename in os.listdir(data_dir):
            if filename.endswith('_chunks.txt'):
                file_path = os.path.join(data_dir, filename)
                print(f"\nProcessing {filename}...")
                embeddings = self.process_chunk_file(file_path)
                all_embeddings[filename] = embeddings
        
        return all_embeddings

if __name__ == "__main__":
    data_dir = "data"
    creator = EmbeddingsCreator()
    embeddings = creator.process_all_chunk_files(data_dir)
    print("\nEmbeddings creation complete!")