import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger("Chunker")

class Chunker:
    """Splits document text into semantic sentence-boundary-preserving chunks with overlap."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Splits English and Hindi text by sentence boundaries (., !, ?, and Hindi danda ।)."""
        if not text:
            return []
        # Split on sentence boundaries, preserving the punctuation in the sentences
        # Using a regex that splits on . ! ? । and keeps them
        sentence_endings = re.compile(r'([^.!?।\n]+[.!?।\n]*)')
        parts = sentence_endings.findall(text)
        
        sentences = []
        for part in parts:
            part_str = part.strip()
            if part_str:
                sentences.append(part_str)
        return sentences

    def chunk_document(self, doc_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunks a list of document pages.
        Maintains page-level metadata and appends chunk-level metadata.
        """
        chunks = []
        
        # Combine all pages but remember the page transitions or just chunk page by page.
        # Chunking page by page is safer for metadata, but combining and splitting is better for flow.
        # Let's combine text, keeping track of page numbers per sentence.
        all_sentences = []
        sentence_metadata = []
        
        for page in doc_pages:
            page_text = page["text"]
            page_meta = page["metadata"]
            
            sentences = self.split_into_sentences(page_text)
            for s in sentences:
                all_sentences.append(s)
                sentence_metadata.append(page_meta)

        if not all_sentences:
            return []

        # Group sentences into chunks
        current_chunk_sentences = []
        current_chunk_len = 0
        
        i = 0
        while i < len(all_sentences):
            sentence = all_sentences[i]
            sent_len = len(sentence)
            
            # If a single sentence is longer than chunk_size, we just force it in
            if sent_len >= self.chunk_size and not current_chunk_sentences:
                current_chunk_sentences.append(sentence)
                self._create_chunk(current_chunk_sentences, sentence_metadata[i], chunks)
                current_chunk_sentences = []
                current_chunk_len = 0
                i += 1
                continue
                
            if current_chunk_len + sent_len > self.chunk_size:
                # Save current chunk
                # Find the primary page of this chunk (usually the page of the first sentence)
                primary_meta = sentence_metadata[i - len(current_chunk_sentences)] if current_chunk_sentences else sentence_metadata[i]
                self._create_chunk(current_chunk_sentences, primary_meta, chunks)
                
                # Backtrack for overlap
                overlap_len = 0
                overlap_sentences = []
                # Go backwards to add overlapping sentences
                j = i - 1
                while j >= 0 and j >= i - len(current_chunk_sentences):
                    back_sent = all_sentences[j]
                    if overlap_len + len(back_sent) <= self.chunk_overlap:
                        overlap_sentences.insert(0, back_sent)
                        overlap_len += len(back_sent)
                        j -= 1
                    else:
                        break
                
                current_chunk_sentences = overlap_sentences
                current_chunk_len = overlap_len
                # Notice we do NOT increment i here, so the current sentence will be processed in the next iteration
            else:
                current_chunk_sentences.append(sentence)
                current_chunk_len += sent_len + 1  # +1 for space/newline
                i += 1
                
        # Flush the last chunk
        if current_chunk_sentences:
            primary_meta = sentence_metadata[i - len(current_chunk_sentences)] if len(sentence_metadata) >= i else (sentence_metadata[-1] if sentence_metadata else {})
            self._create_chunk(current_chunk_sentences, primary_meta, chunks)

        return chunks

    def _create_chunk(self, sentences: List[str], metadata: Dict[str, Any], chunks: List[Dict[str, Any]]):
        chunk_text = " ".join(sentences).strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "metadata": metadata.copy()
            })
