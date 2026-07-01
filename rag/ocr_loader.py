import fitz  # PyMuPDF
import logging
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import io

logger = logging.getLogger("OCRLoader")

class OCRLoader:
    """Loader to perform OCR on scanned PDF files using RapidOCR."""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
            
        # Lazy import of RapidOCR to avoid loading on startup if not needed
        try:
            from rapidocr_onnxruntime import RapidOCR
            self.ocr_engine = RapidOCR()
            logger.info("RapidOCR initialized successfully.")
        except ImportError as e:
            logger.error("rapidocr-onnxruntime is not installed or failed to import.")
            raise e

    def load(self, dpi: int = 150) -> List[Dict[str, Any]]:
        """
        Renders each PDF page as an image and extracts text using RapidOCR.
        Returns a list of dictionaries with text and page metadata.
        """
        pages_content = []
        try:
            doc = fitz.open(self.filepath)
            logger.info(f"Starting OCR for scanned PDF: {self.filepath.name} with {len(doc)} pages.")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Render page to image pixmap
                pix = page.get_pixmap(dpi=dpi)
                img_data = pix.tobytes("png")
                
                # Perform OCR
                result, elapse = self.ocr_engine(img_data)
                
                text_lines = []
                if result:
                    for line in result:
                        # line is typically [[coords], text, confidence]
                        text_lines.append(line[1])
                
                page_text = "\n".join(text_lines)
                logger.info(f"Page {page_num + 1}/{len(doc)} OCR complete in {elapse if isinstance(elapse, (int, float)) else sum(elapse):.2f}s")
                
                pages_content.append({
                    "text": page_text,
                    "metadata": {
                        "source": self.filepath.name,
                        "filepath": str(self.filepath),
                        "page": page_num + 1,
                        "total_pages": len(doc),
                        "ocr_applied": True
                    }
                })
            doc.close()
        except Exception as e:
            logger.error(f"Error performing OCR on {self.filepath}: {e}")
            raise e
            
        return pages_content
