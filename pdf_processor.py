import os
from PyPDF2 import PdfReader

class PDFProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a single PDF file"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return None

    def process_all_pdfs(self):
        """Process all PDFs in the data directory"""
        extracted_texts = {}
        
        for filename in os.listdir(self.data_dir):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(self.data_dir, filename)
                print(f"Processing {filename}...")
                
                text = self.extract_text_from_pdf(pdf_path)
                if text:
                    extracted_texts[filename] = text
                    
                    # Save extracted text to a file
                    txt_filename = os.path.splitext(filename)[0] + '.txt'
                    txt_path = os.path.join(self.data_dir, txt_filename)
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    print(f"Saved extracted text to {txt_filename}")
        
        return extracted_texts

if __name__ == "__main__":
    data_dir = "data"  # relative path to your data directory
    processor = PDFProcessor(data_dir)
    extracted_texts = processor.process_all_pdfs()
    print("\nProcessing complete!")
    print(f"Processed {len(extracted_texts)} PDF files.")