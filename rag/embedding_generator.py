import os
import torch
import logging
from typing import List
from sentence_transformers import SentenceTransformer
from config import MODELS_DIR

logger = logging.getLogger("EmbeddingGenerator")

class EmbeddingGenerator:
    """Generates text embeddings locally using SentenceTransformers, supporting GPU acceleration."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", gpu_enable: bool = True):
        self.model_name = model_name
        self.gpu_enable = gpu_enable
        self.device = "cuda" if (gpu_enable and torch.cuda.is_available()) else "cpu"
        self._model = None

    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading embedding model '{self.model_name}' on device: {self.device}...")
            # Load from local MODELS_DIR or download if not present
            # sentence-transformers can use cache_folder to store models
            self._model = SentenceTransformer(
                self.model_name, 
                device=self.device, 
                cache_folder=str(MODELS_DIR)
            )
            logger.info(f"Embedding model '{self.model_name}' loaded successfully.")
        return self._model

    def _preprocess_texts(self, texts: List[str], is_query: bool = False) -> List[str]:
        """Preprocesses texts based on the model's standard expectations (e.g., E5 prefixes)."""
        # For E5 models, queries need "query: " prefix and documents need "passage: " prefix
        if "e5" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            return [prefix + text for text in texts]
        
        # For BGE models, query usually needs a prefix for retrieval
        if "bge" in self.model_name.lower() and is_query:
            # Standard BGE instruction query prefix
            prefix = "Represent this sentence for searching relevant passages: "
            return [prefix + text for text in texts]
            
        return texts

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of document chunks."""
        if not texts:
            return []
        processed_texts = self._preprocess_texts(texts, is_query=False)
        # Convert to float list of lists
        embeddings = self.model.encode(
            processed_texts, 
            show_progress_bar=False, 
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Generates embedding for a single user query."""
        processed_text = self._preprocess_texts([text], is_query=True)[0]
        embedding = self.model.encode(
            [processed_text], 
            show_progress_bar=False, 
            normalize_embeddings=True
        )[0]
        return embedding.tolist()
