import re
import logging
from typing import List, Dict, Any, Set

logger = logging.getLogger("CitationValidator")

class CitationValidator:
    """Validates if citations, acts, and sections in the LLM response exist in the retrieved context."""

    # Simple patterns to extract citations and sections
    CITATION_PATTERN = re.compile(
        r'\b(?:AIR|SCC|INSC|SCR|SCALE|JT|ILR|CrLJ)\s+\d{4}\s+\S+(?:\s+\d+)?\b|\b\d{4}\s+INSC\s+\d+\b', 
        re.IGNORECASE
    )
    SECTION_PATTERN = re.compile(
        r'\b(?:section|sec\.|u/s)\s*(\d+[A-Z]*)\b', 
        re.IGNORECASE
    )

    def extract_citations(self, text: str) -> Set[str]:
        """Extracts standard Indian legal citation patterns from text."""
        matches = self.CITATION_PATTERN.findall(text)
        return {m.strip().upper() for m in matches}

    def extract_sections(self, text: str) -> Set[str]:
        """Extracts numeric section references from text."""
        matches = self.SECTION_PATTERN.findall(text)
        return {m.strip() for m in matches}

    def validate(self, response_text: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validates citations and sections from the LLM response against context chunks.
        Returns validation results with flags for unverified items.
        """
        # Combine all context text
        context_text = "\n".join(chunk["text"] for chunk in context_chunks)
        
        # Extract from response
        response_citations = self.extract_citations(response_text)
        response_sections = self.extract_sections(response_text)
        
        # Extract from context
        context_citations = self.extract_citations(context_text)
        context_sections = self.extract_sections(context_text)
        
        # Check for unverified citations
        unverified_citations = []
        for citation in response_citations:
            # Check if this exact citation or a substantial part of it is in the context
            # We do a substring search in context to be forgiving of minor spacing differences
            found = False
            for c_cit in context_citations:
                if citation in c_cit or c_cit in citation:
                    found = True
                    break
            if not found and citation not in context_text.upper():
                unverified_citations.append(citation)

        # Check for unverified sections
        unverified_sections = []
        for sec in response_sections:
            if sec not in context_sections and sec not in context_text:
                unverified_sections.append(sec)

        is_valid = len(unverified_citations) == 0 and len(unverified_sections) == 0

        logger.info(f"Citation validation complete. Valid: {is_valid}. "
                    f"Unverified Citations: {unverified_citations}, Unverified Sections: {unverified_sections}")

        return {
            "is_valid": is_valid,
            "unverified_citations": unverified_citations,
            "unverified_sections": unverified_sections
        }
