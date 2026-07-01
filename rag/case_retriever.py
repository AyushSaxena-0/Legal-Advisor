import json
import logging
from typing import List, Dict, Any
from rag.hybrid_retriever import HybridRetriever

logger = logging.getLogger("CaseRetriever")

class CaseRetriever:
    """Retrieves similar court judgments based on the hybrid RAG search."""

    def __init__(self, db_conn, hybrid_retriever: HybridRetriever):
        self.db_conn = db_conn
        self.hybrid_retriever = hybrid_retriever

    def retrieve_similar_judgments(self, query: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves top similar judgments by executing a hybrid search,
        filtering for judgment documents, and pulling full judgment records.
        """
        logger.info(f"Retrieving similar judgments for: '{query}'")
        
        # Get hybrid search results (which includes Cross-Encoder scores)
        # Note: HybridRetriever returns up to 8 chunks. Let's ask for more if needed,
        # or we can write a dedicated query filtering for judgments.
        # But wait, we can reuse HybridRetriever.retrieve() and filter.
        # To make sure we get enough judgments, let's execute a search.
        # Let's temporarily increase retrieval counts or filter inside DB query.
        # Since HybridRetriever searches everything, some chunks will be judgments, some bare acts.
        # Let's check how we can fetch the judgments directly.
        # If we use hybrid_retriever.retrieve(), it returns top 8 chunks.
        # We can extract the judgments from there. Let's do that!
        
        chunks = self.hybrid_retriever.retrieve(query)
        
        # Filter chunks that belong to judgments
        judgment_chunks = [c for c in chunks if c["metadata"].get("is_judgment", False)]
        
        # Group by Case Name or Doc ID to avoid duplicates of the same case
        seen_cases = {}
        for c in judgment_chunks:
            doc_name = c["metadata"].get("filename", "")
            # We can use filename or case_name as unique identifier
            doc_id_query = """
                SELECT d.id FROM documents d WHERE d.filename = ?
            """
            cursor = self.db_conn.cursor()
            cursor.execute(doc_id_query, (doc_name,))
            row = cursor.fetchone()
            if not row:
                continue
            doc_id = row[0]
            
            # Check if we already registered this case, keep the highest score chunk
            score = c.get("rerank_score", c.get("hybrid_score", 0.0))
            if doc_id not in seen_cases or score > seen_cases[doc_id]["score"]:
                seen_cases[doc_id] = {
                    "score": score,
                    "matched_text": c["text"],
                    "metadata": c["metadata"]
                }
                
        results = []
        for doc_id, info in seen_cases.items():
            # Query the judgments table for details
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT case_name, court, bench, judge, judgment_date, citation, acts, sections, ratio_decidendi, important_paragraphs
                FROM judgments
                WHERE doc_id = ?
            """, (doc_id,))
            j_row = cursor.fetchone()
            
            if j_row:
                case_name, court, bench, judge, judgment_date, citation, acts, sections, ratio_decidendi, important_paragraphs = j_row
                
                # If ratio decidendi is empty, fallback to a placeholder or summary
                if not ratio_decidendi:
                    ratio_decidendi = "Ratio Decidendi not explicitly indexed. Refer to judgment text."
                
                # Format response
                results.append({
                    "case_name": case_name or info["metadata"].get("case_name", "Unknown Case"),
                    "court": court or info["metadata"].get("court", "Unknown Court"),
                    "bench": bench or info["metadata"].get("bench", "Unknown Bench"),
                    "judgment_date": judgment_date or info["metadata"].get("judgment_date", "Unknown Date"),
                    "citation": citation or info["metadata"].get("citation", "Unknown Citation"),
                    "applicable_sections": sections or ", ".join(info["metadata"].get("sections", [])),
                    "ratio_decidendi": ratio_decidendi,
                    "important_paragraphs": important_paragraphs or info["matched_text"],
                    "similarity_score": round(info["score"], 4)
                })
            else:
                # Fallback if no specific judgment metadata exists in SQLite, construct from document metadata
                meta = info["metadata"]
                results.append({
                    "case_name": meta.get("case_name", "Unknown Case"),
                    "court": meta.get("court", "Unknown Court"),
                    "bench": meta.get("bench", "Unknown Bench"),
                    "judgment_date": meta.get("judgment_date", "Unknown Date"),
                    "citation": meta.get("citation", "Unknown Citation"),
                    "applicable_sections": ", ".join(meta.get("sections", [])),
                    "ratio_decidendi": "Refer to document content.",
                    "important_paragraphs": info["matched_text"],
                    "similarity_score": round(info["score"], 4)
                })
                
        # Sort by similarity score descending
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results[:top_n]
