import fitz  # PyMuPDF
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger("PDFLoader")

class PDFLoader:
    """Loader to extract text from PDF files using PyMuPDF."""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"PDF file not found: {self.filepath}")

    def load(self) -> List[Dict[str, Any]]:
        """
        Extracts text from PDF page by page.
        Returns a list of dictionaries containing page content and metadata.
        """
        pages_content = []
        try:
            doc = fitz.open(self.filepath)
            logger.info(f"Successfully opened PDF: {self.filepath.name} with {len(doc)} pages.")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                pages_content.append({
                    "text": text,
                    "metadata": {
                        "source": self.filepath.name,
                        "filepath": str(self.filepath),
                        "page": page_num + 1,
                        "total_pages": len(doc)
                    }
                })
            doc.close()
        except Exception as e:
            logger.error(f"Error loading PDF {self.filepath}: {e}")
            raise e
            
        return pages_content

    @staticmethod
    def is_scanned(pages_content: List[Dict[str, Any]], threshold_chars_per_page: int = 50) -> bool:
        """
        Heuristic to determine if the PDF is scanned.
        If the average character count per page is less than threshold, it's likely scanned.
        """
        if not pages_content:
            return True
        total_chars = sum(len(page["text"].strip()) for page in pages_content)
        avg_chars = total_chars / len(pages_content)
        logger.info(f"Average characters per page: {avg_chars:.2f}")
        return avg_chars < threshold_chars_per_page
