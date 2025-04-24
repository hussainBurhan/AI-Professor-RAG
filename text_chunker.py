from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

class TextChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_text(self, text):
        """Split text into chunks"""
        return self.text_splitter.split_text(text)

    def process_text_files(self, data_dir):
        """Process all text files in the directory and create chunks"""
        all_chunks = {}
        
        for filename in os.listdir(data_dir):
            if filename.lower().endswith('.txt'):
                file_path = os.path.join(data_dir, filename)
                print(f"Chunking {filename}...")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    # Create chunks
                    chunks = self.chunk_text(text)
                    all_chunks[filename] = chunks
                    
                    # Save chunks to a new file
                    chunks_filename = os.path.splitext(filename)[0] + '_chunks.txt'
                    chunks_path = os.path.join(data_dir, chunks_filename)
                    
                    with open(chunks_path, 'w', encoding='utf-8') as f:
                        for i, chunk in enumerate(chunks):
                            f.write(f"Chunk {i+1}:\n{chunk}\n{'='*50}\n")
                    
                    print(f"Created {len(chunks)} chunks for {filename}")
                    print(f"Saved chunks to {chunks_filename}")
                
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        
        return all_chunks

if __name__ == "__main__":
    data_dir = "data"  # relative path to your data directory
    chunker = TextChunker()
    chunks = chunker.process_text_files(data_dir)
    print("\nChunking complete!")
    print(f"Processed {len(chunks)} text files.")