import torch
import logging
from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
from config import MODELS_DIR

logger = logging.getLogger("CrossEncoderReranker")

class CrossEncoderReranker:
    """Reranks candidate text chunks relative to a query using a Cross-Encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", gpu_enable: bool = True):
        self.model_name = model_name
        self.gpu_enable = gpu_enable
        self.device = "cuda" if (gpu_enable and torch.cuda.is_available()) else "cpu"
        self._model = None
        self._loaded = False

    @property
    def model(self):
        if not self._loaded:
            logger.info(f"Loading Cross-Encoder model '{self.model_name}' on device: {self.device}...")
            try:
                self._model = CrossEncoder(
                    self.model_name, 
                    device=self.device, 
                    cache_folder=str(MODELS_DIR)
                )
                logger.info("Cross-Encoder loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading Cross-Encoder '{self.model_name}': {e}. Falling back to non-reranked ordering.")
                self._model = None
            self._loaded = True
        return self._model

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Reranks a list of retrieved chunks based on their relevance to the query.
        Returns the top_k re-ranked chunks.
        """
        if not chunks or self.model is None:
            return chunks[:top_k]

        # Prepare pairs: (query, chunk_text)
        pairs = [(query, chunk["text"]) for chunk in chunks]
        
        try:
            # Predict scores
            scores = self.model.predict(pairs, show_progress_bar=False)
            
            # Associate scores with chunks
            for idx, score in enumerate(scores):
                # Save the rerank score in the chunk metadata or direct attribute
                chunks[idx]["rerank_score"] = float(score)
                
            # Sort chunks by rerank score descending
            reranked_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
            
            logger.info(f"Cross-Encoder reranking complete. Re-ranked {len(chunks)} down to {min(top_k, len(chunks))} chunks.")
            return reranked_chunks[:top_k]
            
        except Exception as e:
            logger.error(f"Error during Cross-Encoder reranking: {e}. Returning original ordering.")
            return chunks[:top_k]
