import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger("TextCleaner")

class TextCleaner:
    """Utility to clean raw text from documents (remove headers, footers, and sanitize whitespace)."""

    # Common patterns for headers/footers in legal documents
    PAGE_NUM_PATTERNS = [
        re.compile(r'^\s*page\s+\d+\s+(?:of\s+\d+)?\s*$', re.IGNORECASE),
        re.compile(r'^\s*\d+\s*/\s*\d+\s*$'),
        re.compile(r'^\s*-\s*\d+\s*-\s*$'),
        re.compile(r'^\s*\[\s*\d+\s*\]\s*$'),
        re.compile(r'^\s*\d+\s*$'),  # Lone numbers
    ]
    
    LEGAL_DISCLAIMER_PATTERNS = [
        re.compile(r'downloaded from\s+\w+.*', re.IGNORECASE),
        re.compile(r'http[s]?://\S+', re.IGNORECASE),
        re.compile(r'www\.\S+', re.IGNORECASE),
        re.compile(r'court\s+of\s+judicature', re.IGNORECASE),
    ]

    @classmethod
    def clean_text(cls, text: str) -> str:
        """
        Cleans the input text by:
        1. Splitting into lines.
        2. Removing header-like and footer-like lines.
        3. Sanitizing whitespaces.
        4. Re-joining.
        """
        if not text:
            return ""

        lines = text.splitlines()
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append("")
                continue

            # Check page numbers
            is_page_num = False
            for pattern in cls.PAGE_NUM_PATTERNS:
                if pattern.match(stripped):
                    is_page_num = True
                    break
            if is_page_num:
                continue

            # Check legal portal watermark / download footers
            is_watermark = False
            for pattern in cls.LEGAL_DISCLAIMER_PATTERNS:
                if pattern.search(stripped) and len(stripped) < 80:  # Only remove if it is a short watermarked line
                    is_watermark = True
                    break
            if is_watermark:
                continue

            cleaned_lines.append(line)

        # Reconstruct text
        text = "\n".join(cleaned_lines)

        # Normalize whitespace (replace 3+ newlines with 2, replace tabs with spaces, remove trailing whitespace)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        return text

    @classmethod
    def clean_document_pages(cls, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cleans the text content of each page in the list."""
        cleaned_pages = []
        for page in pages:
            cleaned_text = cls.clean_text(page["text"])
            cleaned_pages.append({
                "text": cleaned_text,
                "metadata": page["metadata"]
            })
        return cleaned_pages
