import logging
import json
from typing import List, Dict, Any, Tuple
from rag.faiss_manager import FAISSManager
from rag.bm25_manager import BM25Manager
from rag.embedding_generator import EmbeddingGenerator
from rag.cross_encoder_reranker import CrossEncoderReranker

logger = logging.getLogger("HybridRetriever")

class HybridRetriever:
    """Combines FAISS semantic search and BM25 keyword search, normalizes, and reranks chunks."""

    def __init__(
        self,
        db_conn,  # SQLite connection or helper
        embedding_generator: EmbeddingGenerator,
        faiss_manager: FAISSManager,
        bm25_manager: BM25Manager,
        reranker: CrossEncoderReranker,
        settings: Dict[str, Any]
    ):
        self.db_conn = db_conn
        self.embedding_generator = embedding_generator
        self.faiss_manager = faiss_manager
        self.bm25_manager = bm25_manager
        self.reranker = reranker
        self.settings = settings

    def _normalize_scores(self, results: List[Tuple[int, float]]) -> Dict[int, float]:
        """Normalizes scores to [0, 1] range using Min-Max scaling."""
        if not results:
            return {}
        
        chunk_ids = [r[0] for r in results]
        scores = [r[1] for r in results]
        
        min_score = min(scores)
        max_score = max(scores)
        
        normalized = {}
        denom = max_score - min_score
        
        if denom > 1e-6:
            for cid, score in results:
                normalized[cid] = (score - min_score) / denom
        else:
            for cid in chunk_ids:
                normalized[cid] = 1.0  # If all scores are equal, assign 1.0
                
        return normalized

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieves the top context chunks for the user query using the hybrid pipeline.
        """
        top_k_faiss = self.settings.get("top_k_faiss", 30)
        top_k_bm25 = self.settings.get("top_k_bm25", 30)
        faiss_weight = self.settings.get("hybrid_weight_faiss", 0.70)
        bm25_weight = 1.0 - faiss_weight

        logger.info(f"Retrieving for query: '{query}'")

        # 1. Semantic Search using FAISS
        query_vector = self.embedding_generator.embed_query(query)
        faiss_results = self.faiss_manager.search(query_vector, top_k=top_k_faiss)
        
        # 2. Keyword Search using BM25
        bm25_results = self.bm25_manager.search(query, top_k=top_k_bm25)

        logger.info(f"FAISS retrieved {len(faiss_results)} candidates. BM25 retrieved {len(bm25_results)} candidates.")

        if not faiss_results and not bm25_results:
            return []

        # 3. Normalize Scores
        faiss_norm = self._normalize_scores(faiss_results)
        bm25_norm = self._normalize_scores(bm25_results)

        # 4. Merge Results & Weighted Scoring
        all_chunk_ids = set(faiss_norm.keys()).union(set(bm25_norm.keys()))
        
        hybrid_scores = []
        for cid in all_chunk_ids:
            f_score = faiss_norm.get(cid, 0.0)
            b_score = bm25_norm.get(cid, 0.0)
            
            h_score = (faiss_weight * f_score) + (bm25_weight * b_score)
            hybrid_scores.append((cid, h_score))

        # Sort and take Top 20
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        top_20_scores = hybrid_scores[:20]
        top_20_ids = [item[0] for item in top_20_scores]

        if not top_20_ids:
            return []

        # 5. Fetch Full Chunks and Metadata from DB
        chunks_map = self._fetch_chunks_from_db(top_20_ids)
        
        # Build candidate list with similarity scores and original ranking info
        candidates = []
        for cid, h_score in top_20_scores:
            if cid in chunks_map:
                chunk = chunks_map[cid]
                # Embed the hybrid score
                chunk["hybrid_score"] = h_score
                chunk["faiss_norm_score"] = faiss_norm.get(cid, 0.0)
                chunk["bm25_norm_score"] = bm25_norm.get(cid, 0.0)
                candidates.append(chunk)

        # 6. Cross Encoder Reranker to get Top 8 Context Chunks
        reranked_chunks = self.reranker.rerank(query, candidates, top_k=8)
        
        return reranked_chunks

    def _fetch_chunks_from_db(self, chunk_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Queries the SQLite database to fetch chunk details and their parent document metadata."""
        if not chunk_ids:
            return {}

        chunks_map = {}
        try:
            cursor = self.db_conn.cursor()
            
            # Parametrized query to prevent injection
            placeholders = ",".join("?" for _ in chunk_ids)
            query = f"""
                SELECT c.id, c.text, c.chunk_index, c.metadata, d.filename, d.filepath, d.is_judgment
                FROM chunks c
                JOIN documents d ON c.doc_id = d.id
                WHERE c.id IN ({placeholders})
            """
            cursor.execute(query, chunk_ids)
            rows = cursor.fetchall()
            
            for row in rows:
                cid, text, c_idx, meta_str, filename, filepath, is_judgment = row
                
                # Parse JSON metadata
                try:
                    meta = json.loads(meta_str) if meta_str else {}
                except Exception:
                    meta = {}
                
                # Add default fields
                meta["filename"] = filename
                meta["filepath"] = filepath
                meta["is_judgment"] = bool(is_judgment)
                
                chunks_map[cid] = {
                    "id": cid,
                    "text": text,
                    "chunk_index": c_idx,
                    "metadata": meta
                }
                
        except Exception as e:
            logger.error(f"Error fetching chunks from SQLite: {e}")
            
        return chunks_map
