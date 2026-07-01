import logging
from typing import List, Dict, Any

logger = logging.getLogger("ResponseFormatter")

class ResponseFormatter:
    """Formats RAG outputs (chunks, cases, acts, sections) into clean Markdown/HTML for the Gradio UI."""

    @staticmethod
    def format_sidebar_cases(cases: List[Dict[str, Any]]) -> str:
        """Formats similar cases into a clean Markdown list for the sidebar."""
        if not cases:
            return "*No matching cases found in retrieved context.*"
        
        md_lines = []
        for i, case in enumerate(cases):
            name = case.get("case_name", "Unknown Case")
            court = case.get("court", "Unknown Court")
            citation = case.get("citation", "No Citation")
            date = case.get("judgment_date", "")
            score = case.get("similarity_score", 0.0)
            
            md_lines.append(
                f"**{i+1}. {name}**\n"
                f"- Court: {court}\n"
                f"- Citation: {citation}\n"
                f"- Date: {date}\n"
                f"- Relevance: {score:.4f}\n"
            )
        return "\n".join(md_lines)

    @staticmethod
    def format_sidebar_metadata(chunks: List[Dict[str, Any]]) -> tuple:
        """Extracts unique Acts and Sections from chunks and formats them as Markdown."""
        acts = set()
        sections = set()
        
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            for act in meta.get("acts", []):
                acts.add(act)
            for sec in meta.get("sections", []):
                sections.add(sec)
                
        # Format Acts
        if acts:
            acts_md = "\n".join(f"- {act}" for act in sorted(acts))
        else:
            acts_md = "*No specific Acts identified in context.*"
            
        # Format Sections
        if sections:
            sections_md = "\n".join(f"- Section {sec}" for sec in sorted(sections, key=lambda x: str(x)))
        else:
            sections_md = "*No specific Sections identified in context.*"
            
        return acts_md, sections_md

    @staticmethod
    def format_bottom_chunks(chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks with similarity scores into collapsible markdown blocks."""
        if not chunks:
            return "*No chunks retrieved.*"
            
        md_lines = []
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "").replace("\n", " ")
            meta = chunk.get("metadata", {})
            filename = meta.get("filename", "Unknown Document")
            page = meta.get("page", "Unknown Page")
            f_score = chunk.get("faiss_norm_score", 0.0)
            b_score = chunk.get("bm25_norm_score", 0.0)
            h_score = chunk.get("hybrid_score", 0.0)
            r_score = chunk.get("rerank_score", None)
            
            score_str = f"FAISS: {f_score:.2f} | BM25: {b_score:.2f} | Hybrid: {h_score:.2f}"
            if r_score is not None:
                score_str += f" | Rerank: {r_score:.4f}"
                
            md_lines.append(
                f"<details>\n"
                f"<summary><b>Chunk {i+1}</b> — <i>Source: {filename} (Page {page})</i> [Score: {score_str}]</summary>\n\n"
                f"```text\n{chunk.get('text', '')}\n```\n"
                f"</details>\n"
            )
            
        return "\n".join(md_lines)
