import os
import pickle
import re
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from rank_bm25 import BM25Okapi
from config import BM25_INDEX_DIR

logger = logging.getLogger("BM25Manager")

class BM25Manager:
    """Manages the BM25 search index, supporting persistence and querying."""

    def __init__(self, index_name: str = "bm25_index"):
        self.index_name = index_name
        self.index_path = BM25_INDEX_DIR / f"{index_name}.pkl"
        self.bm25 = None
        self.chunk_ids = []  # List of SQLite chunk IDs corresponding to index i
        self.load()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenizes English and Hindi text using Unicode-aware word matching."""
        if not text:
            return []
        # \w matches any alphanumeric character, including Unicode (Hindi characters)
        return re.findall(r'\w+', text.lower())

    def load(self):
        """Loads BM25 index from disk if it exists."""
        if self.index_path.exists():
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    self.bm25 = data.get("bm25")
                    self.chunk_ids = data.get("chunk_ids", [])
                logger.info(f"Loaded existing BM25 index from {self.index_path}. Vocabulary size: {len(self.bm25.doc_freqs) if self.bm25 else 0}. Total documents: {len(self.chunk_ids)}")
            except Exception as e:
                logger.error(f"Error loading BM25 index from {self.index_path}: {e}. Initializing empty.")
                self.bm25 = None
                self.chunk_ids = []
        else:
            self.bm25 = None
            self.chunk_ids = []

    def save(self):
        """Saves the BM25 index to disk."""
        if self.bm25 is None:
            logger.warning("No BM25 index built. Nothing to save.")
            return False
        try:
            with open(self.index_path, "wb") as f:
                pickle.dump({
                    "bm25": self.bm25,
                    "chunk_ids": self.chunk_ids
                }, f)
            logger.info(f"Saved BM25 index to {self.index_path}.")
            return True
        except Exception as e:
            logger.error(f"Error saving BM25 index to disk: {e}")
            return False

    def build_index(self, chunks: List[Dict[str, Any]]):
        """
        Builds the BM25 index from a list of chunks.
        Each chunk must have 'id' and 'text'.
        """
        if not chunks:
            logger.warning("Empty chunk list provided for BM25. Clearing index.")
            self.bm25 = None
            self.chunk_ids = []
            if self.index_path.exists():
                self.index_path.unlink()
            return

        corpus_tokenized = []
        self.chunk_ids = []

        for chunk in chunks:
            tokens = self.tokenize(chunk["text"])
            corpus_tokenized.append(tokens)
            self.chunk_ids.append(chunk["id"])

        logger.info(f"Building BM25 index for {len(chunks)} chunks.")
        self.bm25 = BM25Okapi(corpus_tokenized)
        self.save()

    def search(self, query: str, top_k: int = 30) -> List[Tuple[int, float]]:
        """
        Searches the BM25 index for the query.
        Returns a list of tuples: (chunk_id, score).
        """
        if self.bm25 is None or not self.chunk_ids:
            return []

        query_tokens = self.tokenize(query)
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Zip scores with chunk IDs
        id_scores = list(zip(self.chunk_ids, scores))
        
        # Sort by score descending and take top_k
        id_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out scores of 0 or less to ensure relevance
        results = [(int(cid), float(score)) for cid, score in id_scores if score > 0]
        
        return results[:top_k]

    def clear(self):
        """Clears the BM25 index."""
        self.bm25 = None
        self.chunk_ids = []
        if self.index_path.exists():
            try:
                self.index_path.unlink()
            except Exception as e:
                logger.error(f"Error deleting BM25 index file: {e}")
