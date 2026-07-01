import os
import faiss
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
from config import FAISS_INDEX_DIR

logger = logging.getLogger("FAISSManager")

class FAISSManager:
    """Manages the offline FAISS index, supporting incremental updates and deletions."""

    def __init__(self, dimension: int = 768, index_name: str = "faiss_index"):
        self.dimension = dimension
        self.index_name = index_name
        self.index_path = FAISS_INDEX_DIR / f"{index_name}.bin"
        self.index = None
        self.load_or_create()

    def load_or_create(self):
        """Loads index from disk, or creates a new one if it doesn't exist."""
        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                logger.info(f"Loaded existing FAISS index from {self.index_path}. Total vectors: {self.index.ntotal}")
                # Ensure the loaded index is an ID map
                if not isinstance(self.index, (faiss.IndexIDMap, faiss.IndexIDMap2)):
                    logger.warning("Loaded FAISS index is not an ID Map. Re-wrapping.")
                    self.index = faiss.IndexIDMap(self.index)
            except Exception as e:
                logger.error(f"Error reading FAISS index from {self.index_path}: {e}. Creating new index.")
                self.create_new_index()
        else:
            self.create_new_index()

    def create_new_index(self):
        """Creates a new FAISS IndexFlatIP (Inner Product, i.e., Cosine Similarity for normalized vectors)."""
        logger.info(f"Creating a new FAISS index with dimension {self.dimension}.")
        # Use Flat Inner Product (for normalized embeddings)
        base_index = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap(base_index)
        self.save()

    def save(self):
        """Saves the FAISS index to disk."""
        try:
            faiss.write_index(self.index, str(self.index_path))
            logger.info(f"FAISS index saved to {self.index_path}. Total vectors: {self.index.ntotal}")
        except Exception as e:
            logger.error(f"Error saving FAISS index to disk: {e}")

    def add_vectors(self, embeddings: List[List[float]], ids: List[int]):
        """Adds a list of embeddings and their corresponding SQLite chunk IDs to the index."""
        if not embeddings or not ids:
            return
        
        if len(embeddings) != len(ids):
            raise ValueError("Number of embeddings does not match number of IDs.")

        arr_embeds = np.array(embeddings, dtype=np.float32)
        arr_ids = np.array(ids, dtype=np.int64)

        # Ensure FAISS has correct dimension
        if arr_embeds.shape[1] != self.dimension:
            # Recreate index if dimension mismatch occurs due to config change
            logger.warning(f"Embedding dimension mismatch ({arr_embeds.shape[1]} vs {self.dimension}). Recreating index.")
            self.dimension = arr_embeds.shape[1]
            self.create_new_index()

        self.index.add_with_ids(arr_embeds, arr_ids)
        self.save()

    def search(self, query_embedding: List[float], top_k: int = 30) -> List[Tuple[int, float]]:
        """
        Searches the FAISS index for the query embedding.
        Returns a list of tuples: (chunk_id, similarity_score).
        """
        if self.index.ntotal == 0:
            return []

        # query_embedding shape needs to be (1, dimension)
        q_arr = np.array([query_embedding], dtype=np.float32)
        
        # Search
        scores, indices = self.index.search(q_arr, top_k)
        
        results = []
        # scores and indices are of shape (1, top_k)
        for score, idx in zip(scores[0], indices[0]):
            # -1 is returned if there aren't enough elements
            if idx != -1:
                # Cosine similarity is in [-1, 1], convert or keep as is
                results.append((int(idx), float(score)))
                
        return results

    def delete_vectors(self, ids: List[int]) -> int:
        """Deletes vectors with the given IDs from the index. Returns count of removed items."""
        if not ids or self.index.ntotal == 0:
            return 0
        
        arr_ids = np.array(ids, dtype=np.int64)
        try:
            num_removed = self.index.remove_ids(arr_ids)
            logger.info(f"Removed {num_removed} vectors from FAISS index.")
            self.save()
            return num_removed
        except Exception as e:
            logger.error(f"Error removing IDs from FAISS index: {e}")
            return 0

    def clear(self):
        """Clears the FAISS index completely."""
        self.create_new_index()
