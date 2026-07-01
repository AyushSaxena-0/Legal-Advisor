import os
import re
import json
import logging
import sqlite3
import requests
import torch
import gradio as gr
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any, Generator, Optional

from config import (
    BASE_DIR, DATABASE_DIR, DOCUMENTS_DIR, LOGS_DIR, CONFIG_JSON_PATH, 
    settings, load_settings, save_settings
)
from rag.embedding_generator import EmbeddingGenerator
from rag.faiss_manager import FAISSManager
from rag.bm25_manager import BM25Manager
from rag.cross_encoder_reranker import CrossEncoderReranker
from rag.document_manager import DocumentManager
from rag.hybrid_retriever import HybridRetriever
from rag.case_retriever import CaseRetriever
from rag.legal_reasoner import LegalReasoner
from rag.response_formatter import ResponseFormatter

logger = logging.getLogger("LegalAIApp")

# Setup static ffmpeg path wrapper for browser webm audio decoding
try:
    import static_ffmpeg
    logger.info("Initializing static-ffmpeg binaries...")
    static_ffmpeg.add_paths()
    logger.info("static-ffmpeg binaries successfully added to PATH.")
except Exception as ffmpeg_err:
    logger.warning(f"Could not load static-ffmpeg wrapper: {ffmpeg_err}")

# Global variables for lazy initialization of RAG components
_embedding_generator = None
_faiss_manager = None
_bm25_manager = None
_reranker = None
_document_manager = None
_hybrid_retriever = None
_case_retriever = None

# ASR transcription pipeline
_asr_pipeline = None

def get_rag_components():
    """Lazily initializes and caches RAG components to prevent heavy startup delays."""
    global _embedding_generator, _faiss_manager, _bm25_manager, _reranker, _document_manager, _hybrid_retriever, _case_retriever
    
    if _document_manager is None:
        logger.info("Initializing offline RAG components...")
        current_settings = load_settings()
        
        # 1. Embedding Generator
        emb_model = current_settings.get("embedding_model", "BAAI/bge-base-en-v1.5")
        gpu_ok = current_settings.get("gpu_enable", True)
        _embedding_generator = EmbeddingGenerator(model_name=emb_model, gpu_enable=gpu_ok)
        
        # 2. FAISS Index Manager
        # bge-base-en-v1.5 output dim is 768, e5-large-v2 is 1024
        dim = 1024 if "e5" in emb_model.lower() else 768
        _faiss_manager = FAISSManager(dimension=dim)
        
        # 3. BM25 Index Manager
        _bm25_manager = BM25Manager()
        
        # 4. Cross Encoder Reranker
        _reranker = CrossEncoderReranker(gpu_enable=gpu_ok)
        
        # 5. Document Manager
        _document_manager = DocumentManager(
            embedding_generator=_embedding_generator,
            faiss_manager=_faiss_manager,
            bm25_manager=_bm25_manager
        )
        
        # 6. Hybrid Retriever
        _hybrid_retriever = HybridRetriever(
            db_conn=_document_manager.conn,
            embedding_generator=_embedding_generator,
            faiss_manager=_faiss_manager,
            bm25_manager=_bm25_manager,
            reranker=_reranker,
            settings=current_settings
        )
        
        # 7. Case Retriever
        _case_retriever = CaseRetriever(
            db_conn=_document_manager.conn,
            hybrid_retriever=_hybrid_retriever
        )
        
    return _document_manager, _hybrid_retriever, _case_retriever, _embedding_generator, _reranker

def reload_components():
    """Forces re-initialization of RAG components when settings are updated."""
    global _embedding_generator, _faiss_manager, _bm25_manager, _reranker, _document_manager, _hybrid_retriever, _case_retriever
    
    _embedding_generator = None
    _faiss_manager = None
    _bm25_manager = None
    _reranker = None
    _document_manager = None
    _hybrid_retriever = None
    _case_retriever = None
    
    # Re-initialize with new parameters
    get_rag_components()

def transcribe_audio(audio_path: str) -> str:
    """ASR voice-to-text pipeline using transformers."""
    global _asr_pipeline
    if not audio_path:
        return ""
    
    try:
        from transformers import pipeline
        if _asr_pipeline is None:
            logger.info("Initializing offline Whisper Tiny model for ASR...")
            device = 0 if (torch.cuda.is_available() and settings.get("gpu_enable", True)) else -1
            _asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-tiny",
                device=device
            )
        
        logger.info(f"Transcribing audio file: {audio_path}")
        result = _asr_pipeline(audio_path)
        return result.get("text", "")
    except Exception as e:
        logger.error(f"Error during audio transcription: {e}")
        return f"[ASR Transcription Failed: {e}]"

def highlight_text(text: str) -> str:
    """Highlights citations, acts, and sections in legal text using html tags."""
    if not text:
        return ""
    
    # Highlight sections (e.g., Section 302, Section 482 of CrPC)
    # Match Section XXX or Sec. XXX or Sec XXX
    highlighted = re.sub(
        r'\b(section|sec\.|u/s|u/ss)\s+(\d+[A-Z]*)\b',
        r'<mark style="background-color: #fbbf24; color: #000000; padding: 2px 4px; border-radius: 4px; font-weight: bold;">\1 \2</mark>',
        text,
        flags=re.IGNORECASE
    )
    
    # Highlight common acts (e.g. IPC, CrPC, Constitution)
    highlighted = re.sub(
        r'\b(indian penal code|ipc|code of criminal procedure|crpc|indian evidence act|iea|constitution of india|constitution|bharatiya nyaya sanhita|bns|bharatiya nagarik suraksha sanhita|bnss|bharatiya sakshya adhiniyam|bsa)\b',
        r'<mark style="background-color: #60a5fa; color: #000000; padding: 2px 4px; border-radius: 4px; font-weight: bold;">\1</mark>',
        highlighted,
        flags=re.IGNORECASE
    )
    
    # Highlight Indian Legal Citations (e.g., 2023 INSC 123, AIR 2021 SC 432, 10 SCC 122)
    highlighted = re.sub(
        r'\b((?:AIR|SCC|INSC|SCR|SCALE|JT)\s+\d{4}\s+\S+(?:\s+\d+)?|\d{4}\s+INSC\s+\d+)\b',
        r'<mark style="background-color: #34d399; color: #000000; padding: 2px 4px; border-radius: 4px; font-weight: bold;">\1</mark>',
        highlighted,
        flags=re.IGNORECASE
    )
    
    return highlighted

def check_ollama_status() -> Tuple[str, List[str]]:
    """Checks if local Ollama server is running and queries available models."""
    base_url = settings.get("ollama_base_url", "http://localhost:11434")
    model_name = settings.get("ollama_model", "qwen3:8b")
    
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=3)
        if response.status_code == 200:
            models_data = response.json()
            models_list = [m["name"] for m in models_data.get("models", [])]
            is_model_present = model_name in models_list or any(model_name in m for m in models_list)
            
            status_text = "ONLINE"
            status_color = "#10b981" # Green
            
            html = f"""
            <div style='padding: 15px; border-radius: 8px; border: 1px solid {status_color}; background-color: rgba(16, 185, 129, 0.1); color: #e2e8f0; font-family: "Segoe UI", sans-serif;'>
                <h4 style='margin: 0 0 8px 0; color: {status_color}; display: flex; align-items: center;'>
                    <span style='height: 10px; width: 10px; background-color: {status_color}; border-radius: 50%; display: inline-block; margin-right: 8px;'></span>
                    Ollama Server: {status_text}
                </h4>
                <p style='margin: 4px 0;'><b>Endpoint:</b> {base_url}</p>
                <p style='margin: 4px 0;'><b>Target Model:</b> <code>{model_name}</code> ({"Installed" if is_model_present else "<span style='color:#ef4444; font-weight:bold;'>Not Installed - run 'ollama pull " + model_name + "'</span>"})</p>
                <p style='margin: 4px 0;'><b>Available Models:</b> {", ".join([f"<code>{m}</code>" for m in models_list]) if models_list else "None"}</p>
            </div>
            """
            return html, models_list
    except Exception as e:
        logger.error(f"Error querying Ollama server: {e}")
        
    status_text = "OFFLINE"
    status_color = "#ef4444" # Red
    html = f"""
    <div style='padding: 15px; border-radius: 8px; border: 1px solid {status_color}; background-color: rgba(239, 68, 68, 0.1); color: #e2e8f0; font-family: "Segoe UI", sans-serif;'>
        <h4 style='margin: 0 0 8px 0; color: {status_color}; display: flex; align-items: center;'>
            <span style='height: 10px; width: 10px; background-color: {status_color}; border-radius: 50%; display: inline-block; margin-right: 8px;'></span>
            Ollama Server: {status_text}
        </h4>
        <p style='margin: 4px 0;'><b>Endpoint:</b> {base_url}</p>
        <p style='margin: 4px 0; color: {status_color};'><b>Error:</b> Could not reach Ollama. Please launch the Ollama desktop app or run the Ollama background service on host machine.</p>
    </div>
    """
    return html, []

# ====================================================
# GRADIO CALLBACKS
# ====================================================

def run_example_callback(example_text: str):
    """Callback to switch tab and pre-fill advisory query."""
    # Returns (query_text, main_tabs_update)
    return example_text, gr.Tabs(selected="advisor_tab")

def clear_advisor_callback():
    """Clears chatbot and advisor page states."""
    return [], [], "", None, "English", "", "", "", ""

def chatbot_respond_stream(
    user_message: str,
    chat_history: List[Dict[str, str]],
    language: str
) -> Generator[Tuple[str, List[Dict[str, str]], List[Dict[str, str]], str, str, str, str], None, None]:
    """Retrieves context and streams chatbot responses."""
    if not user_message.strip():
        yield "", chat_history, chat_history, "", "", "", ""
        return

    # Add user message to history in role/content format
    new_history = chat_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ""}
    ]
    yield "", new_history, new_history, "Searching database...", "Searching...", "Searching...", "Searching..."

    try:
        # Load component references
        doc_manager, hybrid_retriever, case_retriever, _, _ = get_rag_components()
        
        # 1. Retrieve Context for the user message
        chunks = hybrid_retriever.retrieve(user_message)
        
        if not chunks:
            new_history[-1]["content"] = "Insufficient supporting legal material was found in the indexed database."
            yield "", new_history, new_history, "*No cases*", "*No sections*", "*No acts*", "No chunks."
            return

        # 2. Similar Precedent Cases
        similar_cases = case_retriever.retrieve_similar_judgments(user_message)
        cases_md = ResponseFormatter.format_sidebar_cases(similar_cases)
        
        # 3. Sidebar Acts/Sections & Accordion
        acts_md, sections_md = ResponseFormatter.format_sidebar_metadata(chunks)
        chunks_md = ResponseFormatter.format_bottom_chunks(chunks)
        
        yield "", new_history, new_history, cases_md, sections_md, acts_md, chunks_md

        # 4. Stream from Ollama passing the history list
        reasoner = LegalReasoner(
            ollama_url=settings.get("ollama_base_url", "http://localhost:11434"),
            model_name=settings.get("ollama_model", "qwen3:8b")
        )
        
        response_generator = reasoner.generate_response_stream(
            query=user_message,
            chunks=chunks,
            language=language,
            temperature=settings.get("temperature", 0.2),
            context_length=settings.get("context_length", 8192),
            chat_history=chat_history
        )
        
        for partial_text in response_generator:
            new_history[-1]["content"] = partial_text
            yield "", new_history, new_history, cases_md, sections_md, acts_md, chunks_md
            
    except Exception as e:
        logger.error(f"Critical error in chatbot stream: {e}")
        new_history[-1]["content"] = f"An error occurred in RAG pipeline: {e}"
        yield "", new_history, new_history, "", "", "", ""

def search_cases_callback(
    case_name: str, citation: str, court: str, judge: str, 
    year: str, act: str, section: str, keyword: str
) -> Tuple[pd.DataFrame, gr.State, str]:
    """Searches SQLite database for judgments and formats results."""
    db_path = DATABASE_DIR / "legal_advisor.db"
    if not db_path.exists():
        return pd.DataFrame(), [], "Database does not exist. Please upload documents in Knowledge Base tab."

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        query = """
            SELECT d.id, j.case_name, j.court, j.bench, j.judge, j.judgment_date, j.citation, j.acts, j.sections, j.keywords, j.ratio_decidendi, j.important_paragraphs
            FROM judgments j
            JOIN documents d ON j.doc_id = d.id
            WHERE 1=1
        """
        params = []
        
        if case_name:
            query += " AND j.case_name LIKE ?"
            params.append(f"%{case_name}%")
        if citation:
            query += " AND j.citation LIKE ?"
            params.append(f"%{citation}%")
        if court:
            query += " AND j.court LIKE ?"
            params.append(f"%{court}%")
        if judge:
            query += " AND j.judge LIKE ?"
            params.append(f"%{judge}%")
        if year:
            try:
                query += " AND j.year = ?"
                params.append(int(year))
            except ValueError:
                pass
        if act:
            query += " AND j.acts LIKE ?"
            params.append(f"%{act}%")
        if section:
            query += " AND j.sections LIKE ?"
            params.append(f"%{section}%")
        if keyword:
            query += " AND (j.keywords LIKE ? OR j.ratio_decidendi LIKE ?)"
            params.append(f"%{keyword}%")
            params.append(f"%{keyword}%")
            
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return pd.DataFrame(), [], "No judgments matched your search parameters."

        # Map rows to structure
        data_list = []
        df_rows = []
        for r in rows:
            # Columns: id, case_name, court, bench, judge, judgment_date, citation, acts, sections, keywords, ratio, paragraphs
            data_list.append({
                "doc_id": r[0],
                "case_name": r[1],
                "court": r[2],
                "bench": r[3],
                "judge": r[4],
                "judgment_date": r[5],
                "citation": r[6],
                "acts": r[7],
                "sections": r[8],
                "keywords": r[9],
                "ratio_decidendi": r[10],
                "important_paragraphs": r[11]
            })
            
            df_rows.append([
                r[1], # Case Name
                r[2], # Court
                r[6], # Citation
                r[5]  # Date
            ])

        df = pd.DataFrame(df_rows, columns=["Case Name", "Court", "Citation", "Date"])
        return df, data_list, f"Found {len(df)} matching case(s). Select a row to view full details."
    except Exception as e:
        logger.error(f"Search error: {e}")
        return pd.DataFrame(), [], f"Search failed: {e}"

def show_case_details_callback(evt: gr.SelectData, state_data: List[Dict[str, Any]]) -> str:
    """Formats detailed view of selected case in search results."""
    if not state_data or not evt.index:
        return "Select a case row from the table to view details."
        
    row_idx = evt.index[0]
    if row_idx >= len(state_data):
        return "Error loading details."
        
    case = state_data[row_idx]
    
    html = f"""
    <div style='background-color: #1e293b; padding: 20px; border-radius: 8px; color: #f1f5f9; line-height: 1.6;'>
        <h3 style='color:#60a5fa; margin-top:0; border-bottom: 1px solid #475569; padding-bottom: 8px;'>{case["case_name"]}</h3>
        <table style='width:100%; border-collapse:collapse; margin-bottom:15px;'>
            <tr><td style='width:150px; font-weight:bold; color:#94a3b8;'>Court:</td><td>{case["court"]} ({case["bench"] or "N/A"})</td></tr>
            <tr><td style='font-weight:bold; color:#94a3b8;'>Judge(s):</td><td>{case["judge"]}</td></tr>
            <tr><td style='font-weight:bold; color:#94a3b8;'>Date:</td><td>{case["judgment_date"]}</td></tr>
            <tr><td style='font-weight:bold; color:#94a3b8;'>Citation:</td><td><code>{case["citation"]}</code></td></tr>
            <tr><td style='font-weight:bold; color:#94a3b8;'>Acts Cited:</td><td>{case["acts"]}</td></tr>
            <tr><td style='font-weight:bold; color:#94a3b8;'>Sections Cited:</td><td>{case["sections"]}</td></tr>
        </table>
        
        <h4 style='color:#34d399; margin-bottom:5px; border-bottom: 1px solid #475569;'>Ratio Decidendi (Reason for Decision)</h4>
        <p style='white-space:pre-wrap; margin-bottom: 15px;'>{case["ratio_decidendi"] or "Ratio decidendi text not indexed."}</p>
        
        <h4 style='color:#fbbf24; margin-bottom:5px; border-bottom: 1px solid #475569;'>Key Paragraphs</h4>
        <div style='background-color: #0f172a; padding: 12px; border-radius: 4px; border-left: 4px solid #fbbf24; white-space:pre-wrap; font-size:14px; max-height: 250px; overflow-y: auto;'>
            {case["important_paragraphs"] or "No specific paragraphs marked."}
        </div>
    </div>
    """
    return html

def explorer_load_judgments(court: str, year: str, judge: str, bench: str, act: str, section: str) -> Tuple[pd.DataFrame, gr.State, str]:
    """Filters database and returns data for Judgment Explorer."""
    db_path = DATABASE_DIR / "legal_advisor.db"
    if not db_path.exists():
        return pd.DataFrame(), [], "Database is empty."

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        query = """
            SELECT d.id, j.case_name, j.court, j.bench, j.judge, j.judgment_date, j.citation
            FROM judgments j
            JOIN documents d ON j.doc_id = d.id
            WHERE 1=1
        """
        params = []
        
        if court and court != "All":
            query += " AND j.court LIKE ?"
            params.append(f"%{court}%")
        if year:
            try:
                query += " AND j.year = ?"
                params.append(int(year))
            except ValueError:
                pass
        if judge:
            query += " AND j.judge LIKE ?"
            params.append(f"%{judge}%")
        if bench:
            query += " AND j.bench LIKE ?"
            params.append(f"%{bench}%")
        if act:
            query += " AND j.acts LIKE ?"
            params.append(f"%{act}%")
        if section:
            query += " AND j.sections LIKE ?"
            params.append(f"%{section}%")
            
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return pd.DataFrame(), [], "No judgments found matching current filters."

        state_list = []
        df_rows = []
        for r in rows:
            state_list.append({"doc_id": r[0], "case_name": r[1], "citation": r[6]})
            df_rows.append([r[1], r[2], r[6], r[5]])

        df = pd.DataFrame(df_rows, columns=["Case Name", "Court", "Citation", "Date"])
        return df, state_list, f"Loaded {len(df)} judgment(s)."
    except Exception as e:
        logger.error(f"Explorer query error: {e}")
        return pd.DataFrame(), [], f"Failed: {e}"

def explorer_view_full_judgment(evt: gr.SelectData, state_data: List[Dict[str, Any]]) -> str:
    """Combines chunks to render full judgment text with highlighted sections."""
    if not state_data or not evt.index:
        return "Select a judgment from the list to display text."

    row_idx = evt.index[0]
    if row_idx >= len(state_data):
        return "Error finding case."

    doc_id = state_data[row_idx]["doc_id"]
    
    db_path = DATABASE_DIR / "legal_advisor.db"
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Load Chunks
        cursor.execute("SELECT text FROM chunks WHERE doc_id = ? ORDER BY chunk_index ASC", (doc_id,))
        rows = cursor.fetchall()
        
        # Load Judgment Details
        cursor.execute("SELECT case_name, court, citation, judgment_date, judge FROM judgments WHERE doc_id = ?", (doc_id,))
        j_row = cursor.fetchone()
        conn.close()
        
        if not rows:
            return "No text chunks found for this document."

        full_text = "\n\n".join(r[0] for r in rows)
        
        # Details
        case_name = j_row[0] if j_row else "Unknown Case"
        court = j_row[1] if j_row else "Unknown Court"
        citation = j_row[2] if j_row else "No Citation"
        date = j_row[3] if j_row else "Unknown Date"
        judge = j_row[4] if j_row else "Unknown"

        highlighted_text = highlight_text(full_text)

        html = f"""
        <div style='background-color: #0f172a; padding: 25px; border-radius: 8px; border: 1px solid #334155; max-height: 700px; overflow-y: auto; color: #cbd5e1; font-family: "Courier New", Courier, monospace;'>
            <div style='text-align: center; border-bottom: 2px double #475569; padding-bottom: 15px; margin-bottom: 20px;'>
                <h2 style='color:#38bdf8; margin: 0;'>{case_name.upper()}</h2>
                <h3 style='color:#94a3b8; margin: 5px 0;'>IN THE {court.upper()}</h3>
                <p style='margin: 4px 0;'><b>Citation:</b> {citation} | <b>Date:</b> {date}</p>
                <p style='margin: 4px 0;'><b>Hon'ble Judge(s):</b> {judge}</p>
            </div>
            <div style='white-space: pre-wrap; font-size: 15px; line-height: 1.8;'>
{highlighted_text}
            </div>
        </div>
        """
        return html
    except Exception as e:
        logger.error(f"Error loading full judgment text: {e}")
        return f"Error loading full judgment text: {e}"

def knowledge_base_stats() -> Tuple[str, pd.DataFrame]:
    """Gathers SQLite stats & document list for display."""
    doc_manager, _, _, _, _ = get_rag_components()
    stats = doc_manager.get_statistics()
    docs = doc_manager.get_all_documents()
    
    # Format Stats HTML
    stats_html = f"""
    <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; font-family: sans-serif;'>
        <div style='background-color: #1e293b; padding: 15px; border-radius: 6px; text-align: center; border: 1px solid #334155;'>
            <div style='font-size: 24px; font-weight: bold; color: #60a5fa;'>{stats["indexed_documents"]}</div>
            <div style='color: #94a3b8; font-size: 13px; margin-top: 4px;'>Indexed Documents</div>
        </div>
        <div style='background-color: #1e293b; padding: 15px; border-radius: 6px; text-align: center; border: 1px solid #334155;'>
            <div style='font-size: 24px; font-weight: bold; color: #34d399;'>{stats["judgments"]}</div>
            <div style='color: #94a3b8; font-size: 13px; margin-top: 4px;'>Court Judgments</div>
        </div>
        <div style='background-color: #1e293b; padding: 15px; border-radius: 6px; text-align: center; border: 1px solid #334155;'>
            <div style='font-size: 24px; font-weight: bold; color: #c084fc;'>{stats["total_chunks"]}</div>
            <div style='color: #94a3b8; font-size: 13px; margin-top: 4px;'>Total Chunks</div>
        </div>
        <div style='background-color: #1e293b; padding: 15px; border-radius: 6px; text-align: center; border: 1px solid #334155;'>
            <div style='font-size: 24px; font-weight: bold; color: #fbbf24;'>{stats["acts_count"]}</div>
            <div style='color: #94a3b8; font-size: 13px; margin-top: 4px;'>Acts Covered</div>
        </div>
    </div>
    """
    
    # Format Document Table
    df_rows = []
    for doc in docs:
        size_kb = doc["file_size"] / 1024
        df_rows.append([
            doc["id"],
            doc["filename"],
            doc["file_type"].upper(),
            f"{size_kb:.1f} KB",
            doc["chunk_count"],
            "Judgment" if doc["is_judgment"] else "Reference/Act",
            doc["uploaded_at"]
        ])
    
    df = pd.DataFrame(df_rows, columns=["ID", "Filename", "Type", "Size", "Chunks", "Category", "Uploaded At"])
    return stats_html, df

def knowledge_base_ingest(files: List[Any], is_judgment: bool, progress=gr.Progress(track_tqdm=True)) -> Tuple[str, str, pd.DataFrame]:
    """Ingests uploaded files through DocumentManager."""
    if not files:
        return "Please upload files before triggering ingestion.", *knowledge_base_stats()

    doc_manager, _, _, _, _ = get_rag_components()
    success_count = 0
    errors = []
    
    total_files = len(files)
    for i, file_obj in enumerate(files):
        try:
            # Copy to project documents folder
            src_path = Path(file_obj.name)
            dest_path = DOCUMENTS_DIR / src_path.name
            
            # Write bytes to target destination
            with open(dest_path, "wb") as f:
                with open(src_path, "rb") as sf:
                    f.write(sf.read())
            
            # Progress callback wrapper
            def prog_cb(pct, msg):
                step_pct = (i + pct) / total_files
                progress(step_pct, f"[{i+1}/{total_files}] Ingesting {src_path.name}: {msg}")
            
            # Ingest
            doc_manager.ingest_document(dest_path, is_judgment=is_judgment, progress_callback=prog_cb)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to ingest {file_obj.name}: {e}")
            errors.append(f"{src_path.name}: {e}")

    result_msg = f"Successfully indexed {success_count}/{total_files} file(s)."
    if errors:
        result_msg += f"\nErrors:\n" + "\n".join(errors)
        
    stats_html, df = knowledge_base_stats()
    return result_msg, stats_html, df

def knowledge_base_delete(doc_id_str: str) -> Tuple[str, str, pd.DataFrame]:
    """Deletes selected document ID from system."""
    if not doc_id_str:
        return "Please specify a valid Document ID.", *knowledge_base_stats()
        
    try:
        doc_id = int(doc_id_str)
        doc_manager, _, _, _, _ = get_rag_components()
        doc_manager.delete_document(doc_id)
        msg = f"Successfully deleted Document ID {doc_id}."
    except Exception as e:
        logger.error(f"Failed to delete document {doc_id_str}: {e}")
        msg = f"Deletion failed: {e}"
        
    stats_html, df = knowledge_base_stats()
    return msg, stats_html, df

def knowledge_base_rebuild(index_type: str, progress=gr.Progress()) -> Tuple[str, str, pd.DataFrame]:
    """Forces manual index rebuilds."""
    doc_manager, _, _, _, _ = get_rag_components()
    try:
        if index_type == "FAISS":
            progress(0.1, "Initializing re-embedding processes...")
            doc_manager.rebuild_all_indices(progress_callback=progress)
            msg = "Rebuilt FAISS (and synced BM25) successfully!"
        else:
            progress(0.2, "Re-indexing BM25 corpus...")
            doc_manager.rebuild_bm25_index()
            progress(1.0, "Complete")
            msg = "Rebuilt BM25 index successfully!"
    except Exception as e:
        logger.error(f"Rebuild index error: {e}")
        msg = f"Rebuild failed: {e}"
        
    stats_html, df = knowledge_base_stats()
    return msg, stats_html, df

def analytics_details() -> Tuple[str, str]:
    """Populates charts/graphs for Analytics page."""
    # 1. Fetch system statistics
    doc_manager, _, _, _, _ = get_rag_components()
    stats = doc_manager.get_statistics()
    
    # 2. Query Ollama status
    ollama_html, _ = check_ollama_status()
    
    # 3. Format Stats HTML card list
    emb_model = settings.get("embedding_model", "BAAI/bge-base-en-v1.5")
    gpu_status = "ENABLED (CUDA)" if (settings.get("gpu_enable") and torch.cuda.is_available()) else "DISABLED (CPU)"
    
    analytics_html = f"""
    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; font-family: sans-serif; color:#cbd5e1;'>
        <div style='background-color: #1e293b; padding: 20px; border-radius: 8px; border: 1px solid #334155;'>
            <h4 style='color: #60a5fa; margin: 0 0 10px 0;'>DATABASE STORAGE</h4>
            <ul style='padding-left: 20px; margin: 0; line-height: 1.6;'>
                <li><b>Total Files:</b> {stats["indexed_documents"]}</li>
                <li><b>Judgments:</b> {stats["judgments"]}</li>
                <li><b>Total Text Chunks:</b> {stats["total_chunks"]}</li>
            </ul>
        </div>
        <div style='background-color: #1e293b; padding: 20px; border-radius: 8px; border: 1px solid #334155;'>
            <h4 style='color: #34d399; margin: 0 0 10px 0;'>VECTOR STORE & ACCELERATION</h4>
            <ul style='padding-left: 20px; margin: 0; line-height: 1.6;'>
                <li><b>Embedding Model:</b> <code>{emb_model}</code></li>
                <li><b>FAISS Vectors indexed:</b> {stats["faiss_size"]}</li>
                <li><b>Hardware Acceleration:</b> <code>{gpu_status}</code></li>
            </ul>
        </div>
        <div style='background-color: #1e293b; padding: 20px; border-radius: 8px; border: 1px solid #334155;'>
            <h4 style='color: #c084fc; margin: 0 0 10px 0;'>KEYWORD CORPUS</h4>
            <ul style='padding-left: 20px; margin: 0; line-height: 1.6;'>
                <li><b>BM25 Vocabulary Size:</b> {stats["bm25_vocab_size"]} terms</li>
                <li><b>Acts Catalogued:</b> {stats["acts_count"]} Acts</li>
                <li><b>Database Path:</b> <code>{doc_manager.db_path.name}</code></li>
            </ul>
        </div>
    </div>
    """
    
    return analytics_html, ollama_html

def settings_save_callback(
    model: str, chunk_size: float, chunk_overlap: float, 
    faiss_k: float, bm25_k: float, weight: float,
    temp: float, ctx_len: float, gpu: bool, 
    ollama_model: str, ollama_url: str
) -> str:
    """Updates settings JSON and re-initializes models if model path changed."""
    global settings
    
    old_model = settings.get("embedding_model")
    old_gpu = settings.get("gpu_enable")
    
    updated_settings = {
        "embedding_model": model,
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "top_k_faiss": int(faiss_k),
        "top_k_bm25": int(bm25_k),
        "hybrid_weight_faiss": float(weight),
        "temperature": float(temp),
        "context_length": int(ctx_len),
        "gpu_enable": gpu,
        "ollama_model": ollama_model,
        "ollama_base_url": ollama_url
    }
    
    save_settings(updated_settings)
    
    # Reload the globally stored settings
    settings.update(updated_settings)
    
    # Re-initialize models if parameters affecting initialization were changed
    if old_model != model or old_gpu != gpu:
        try:
            logger.info("Settings change requires model reload. Re-loading...")
            reload_components()
            return "Settings saved and RAG Models re-loaded successfully!"
        except Exception as e:
            logger.error(f"Error reloading components: {e}")
            return f"Settings saved, but reloading embedding models failed: {e}"
            
    # For parameters that only affect queries, just update local references in retriever
    try:
        _, hybrid_retriever, _, _, _ = get_rag_components()
        hybrid_retriever.settings = settings
    except Exception:
        pass
        
    return "Settings saved successfully!"

# ====================================================
# GRADIO INTERFACE
# ====================================================

# Premium stylesheet for custom dark mode design and glassmorphism elements
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background-color: #09090b !important;
    background-image: radial-gradient(circle at top center, rgba(99, 102, 241, 0.12), transparent 55%) !important;
}

h1, h2, h3, h4, .logo-text {
    font-family: 'Outfit', sans-serif !important;
}

/* Glassmorphism block cards */
.gradio-container .block {
    background: rgba(24, 24, 27, 0.45) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(63, 63, 70, 0.3) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.gradio-container .block:hover {
    border-color: rgba(99, 102, 241, 0.35) !important;
    box-shadow: 0 8px 32px 0 rgba(99, 102, 241, 0.08) !important;
}

/* Premium bubble styling */
.chat-message.user {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    border-radius: 18px 18px 2px 18px !important;
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.25) !important;
    color: white !important;
}

.chat-message.bot {
    background: #18181b !important;
    border: 1px solid #27272a !important;
    border-radius: 18px 18px 18px 2px !important;
    color: #e4e4e7 !important;
}

/* Action button styling */
.gradio-container button.primary {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25) !important;
}

.gradio-container button.primary:hover {
    transform: translateY(-1.5px) !important;
    box-shadow: 0 6px 18px rgba(79, 70, 229, 0.4) !important;
    filter: brightness(1.1);
}

.gradio-container button.secondary {
    background: rgba(39, 39, 42, 0.6) !important;
    border: 1px solid rgba(63, 63, 70, 0.8) !important;
    color: #a1a1aa !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}

.gradio-container button.secondary:hover {
    background: #27272a !important;
    border-color: #6366f1 !important;
    color: white !important;
}

/* Accordion visual enhancements */
.gradio-container .accordion {
    border-radius: 14px !important;
    border: 1px solid rgba(63, 63, 70, 0.3) !important;
    background: rgba(24, 24, 27, 0.4) !important;
}
"""

# Dark theme customization
dark_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="zinc"
).set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_color="*neutral_800",
    input_background_fill="*neutral_950",
    input_border_color="*neutral_800",
    button_secondary_background_fill="*neutral_800",
)

with gr.Blocks(css=custom_css, title="NYAYA MITRA: Indian AI Legal Advisor") as demo:
    
    # Premium Header Banner with gradient borders and glowing icon
    gr.HTML("""
    <div style='display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 25px; margin-bottom: 25px; border-radius: 16px; background: linear-gradient(135deg, rgba(99,102,241,0.04) 0%, rgba(139,92,246,0.04) 100%); border: 1px solid rgba(99,102,241,0.15); box-shadow: 0 8px 32px 0 rgba(0,0,0,0.37); backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);'>
        <div style='display: flex; align-items: center; gap: 15px; margin-bottom: 5px;'>
            <span style='font-size: 36px; filter: drop-shadow(0 0 10px rgba(99,102,241,0.6));'>⚖️</span>
            <h1 style='color: white; margin: 0; font-size: 34px; font-weight: 900; letter-spacing: 2px; background: linear-gradient(to right, #6366f1, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>NYAYA MITRA</h1>
        </div>
        <p style='color: #a1a1aa; font-size: 15px; margin: 0; font-weight: 400; letter-spacing: 0.5px;'>Offline Retrieval-Augmented Indian Legal Intelligence System</p>
    </div>
    """)
    chat_history_state = gr.State(value=[])
    
    with gr.Row():
        # Left Column: Chat & Input
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Nyaya Mitra Legal Assistant", 
                height=550
            )
            
            with gr.Group():
                with gr.Row():
                    query_input = gr.Textbox(
                        label="Ask a question or describe your situation", 
                        placeholder="Type here... e.g. The police are illegally detaining me.",
                        scale=4
                    )
                    lang_selector = gr.Dropdown(
                        choices=["English", "Hindi"], 
                        value="English", 
                        label="Language",
                        scale=1
                    )
                
                with gr.Row():
                    voice_input = gr.Audio(
                        label="Record voice question", 
                        sources=["microphone"], 
                        type="filepath",
                        scale=3
                    )
                    with gr.Column(scale=2):
                        with gr.Row():
                            clear_btn = gr.Button("Clear Chat", variant="stop")
                            submit_btn = gr.Button("Send", variant="primary")
            
            gr.Markdown("### 💡 Quick Templates / Example Situations")
            with gr.Row():
                with gr.Column(scale=1):
                    ex1 = gr.Button("The police are illegally detaining me.", variant="secondary")
                    ex2 = gr.Button("My landlord is threatening me.", variant="secondary")
                    ex3 = gr.Button("My employer has not paid my salary.", variant="secondary")
                with gr.Column(scale=1):
                    ex4 = gr.Button("My wife filed a false complaint.", variant="secondary")
                    ex5 = gr.Button("My property has been occupied illegally.", variant="secondary")
                    ex6 = gr.Button("My phone was stolen.", variant="secondary")
            
        # Right Column: Sidebar (Metadata)
        with gr.Column(scale=2, min_width=300):
            with gr.Group():
                gr.Markdown("### 🏛️ Retrieved Cases")
                sidebar_cases = gr.Markdown(value="*None*")
                
            with gr.Group():
                gr.Markdown("### 📄 Relevant Sections")
                sidebar_sections = gr.Markdown(value="*None*")
                
            with gr.Group():
                gr.Markdown("### 📜 Relevant Acts")
                sidebar_acts = gr.Markdown(value="*None*")
    
    # Bottom Accordion: Retrieved Chunks
    with gr.Accordion("🔍 Retrieved Chunks, Scores, and Sources", open=False):
        bottom_chunks = gr.Markdown(value="*None*")

    # ====================================================
    # BINDINGS AND ROUTING
    # ====================================================

    # Bind examples directly to the chatbot stream
    for ex_btn in [ex1, ex2, ex3, ex4, ex5, ex6]:
        ex_btn.click(
            fn=chatbot_respond_stream,
            inputs=[ex_btn, chat_history_state, lang_selector],
            outputs=[query_input, chatbot, chat_history_state, sidebar_cases, sidebar_sections, sidebar_acts, bottom_chunks]
        )

    # Bind voice input change to auto-transcribe into the textbox
    voice_input.change(
        fn=transcribe_audio,
        inputs=[voice_input],
        outputs=[query_input]
    )

    # Bind clear advisor
    clear_btn.click(
        fn=clear_advisor_callback,
        inputs=[],
        outputs=[chatbot, chat_history_state, query_input, voice_input, lang_selector, sidebar_cases, sidebar_sections, sidebar_acts, bottom_chunks]
    )

    # Bind chatbot message submit
    submit_btn.click(
        fn=chatbot_respond_stream,
        inputs=[query_input, chat_history_state, lang_selector],
        outputs=[query_input, chatbot, chat_history_state, sidebar_cases, sidebar_sections, sidebar_acts, bottom_chunks]
    )

if __name__ == "__main__":
    # Lazy initial components build on run
    try:
        get_rag_components()
    except Exception as startup_err:
        logger.error(f"Startup warning: models will be loaded on demand in browser: {startup_err}")
        
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=dark_theme, css="body { background-color: #09090b; }")
