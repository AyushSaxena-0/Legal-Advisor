import os
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import xml.etree.ElementTree as ET
import zipfile

from config import DATABASE_DIR, DOCUMENTS_DIR, settings
from rag.pdf_loader import PDFLoader
from rag.ocr_loader import OCRLoader
from rag.text_cleaner import TextCleaner
from rag.chunker import Chunker
from rag.metadata_extractor import MetadataExtractor
from rag.embedding_generator import EmbeddingGenerator
from rag.faiss_manager import FAISSManager
from rag.bm25_manager import BM25Manager

logger = logging.getLogger("DocumentManager")

class DocumentManager:
    """Manages document ingestion, storage in SQLite, embedding updates, and index sync."""

    def __init__(
        self,
        db_path: Path = DATABASE_DIR / "legal_advisor.db",
        embedding_generator: Optional[EmbeddingGenerator] = None,
        faiss_manager: Optional[FAISSManager] = None,
        bm25_manager: Optional[BM25Manager] = None
    ):
        self.db_path = db_path
        self.embedding_generator = embedding_generator
        self.faiss_manager = faiss_manager
        self.bm25_manager = bm25_manager
        
        # Connect and initialize schema
        self.conn = self._get_db_connection()
        self._initialize_schema()

    def _get_db_connection(self):
        """Returns a connection to SQLite database, enabling foreign key constraints."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _initialize_schema(self):
        """Creates tables for documents, chunks, and judgments if they do not exist."""
        cursor = self.conn.cursor()
        
        # Documents Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE,
                filepath TEXT,
                file_type TEXT,
                file_size INTEGER,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                chunk_count INTEGER DEFAULT 0,
                is_judgment BOOLEAN DEFAULT 0
            );
        """)
        
        # Chunks Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER,
                chunk_index INTEGER,
                text TEXT,
                metadata TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
            );
        """)
        
        # Judgments Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS judgments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER UNIQUE,
                case_name TEXT,
                court TEXT,
                bench TEXT,
                judge TEXT,
                judgment_date TEXT,
                citation TEXT,
                acts TEXT,
                sections TEXT,
                keywords TEXT,
                state TEXT,
                court_level TEXT,
                year INTEGER,
                ratio_decidendi TEXT,
                important_paragraphs TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
            );
        """)
        
        self.conn.commit()
        logger.info("SQLite database schema initialized successfully.")

    def parse_file_to_text(self, filepath: Path) -> List[Dict[str, Any]]:
        """Parses various file types and returns a list of pages (dict with 'text' and 'metadata')."""
        ext = filepath.suffix.lower()
        filename = filepath.name
        
        if ext == ".pdf":
            # Heuristically check if scanned, run OCR if so
            loader = PDFLoader(str(filepath))
            pages = loader.load()
            if PDFLoader.is_scanned(pages):
                logger.info(f"PDF {filename} detected as scanned. Delegating to OCR Loader.")
                ocr_loader = OCRLoader(str(filepath))
                pages = ocr_loader.load()
            return pages
            
        elif ext in [".txt", ".md", ".markdown"]:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return [{"text": content, "metadata": {"source": filename, "filepath": str(filepath), "page": 1, "total_pages": 1}}]
            
        elif ext == ".json":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
            content = json.dumps(data, indent=2, ensure_ascii=False)
            return [{"text": content, "metadata": {"source": filename, "filepath": str(filepath), "page": 1, "total_pages": 1}}]
            
        elif ext == ".csv":
            import csv
            text_lines = []
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    text_lines.append(f"Row {i+1}: " + ", ".join(row))
            content = "\n".join(text_lines)
            return [{"text": content, "metadata": {"source": filename, "filepath": str(filepath), "page": 1, "total_pages": 1}}]
            
        elif ext == ".docx":
            content = ""
            try:
                import docx
                doc = docx.Document(filepath)
                content = "\n".join([p.text for p in doc.paragraphs])
            except ImportError:
                # Fallback zip-xml parsing
                logger.info(f"python-docx not installed. Using zip fallback for {filename}.")
                texts = []
                try:
                    with zipfile.ZipFile(filepath) as docx_zip:
                        xml_content = docx_zip.read('word/document.xml')
                        root = ET.fromstring(xml_content)
                        for paragraph in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
                            texts.append(paragraph.text)
                    content = "\n".join(texts)
                except Exception as zip_err:
                    logger.error(f"Error in zip fallback parsing for docx: {zip_err}")
                    raise zip_err
            return [{"text": content, "metadata": {"source": filename, "filepath": str(filepath), "page": 1, "total_pages": 1}}]
            
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def ingest_document(self, filepath: Path, is_judgment: bool = False, progress_callback=None) -> Dict[str, Any]:
        """Ingests a new document, extracts text/metadata, chunks, embeds, and indexes."""
        filename = filepath.name
        file_size = filepath.stat().st_size
        file_type = filepath.suffix.lower()[1:]
        
        logger.info(f"Starting ingestion for: {filename}")
        if progress_callback: progress_callback(0.1, "Extracting text from file...")

        # 1. Parse file
        pages = self.parse_file_to_text(filepath)
        
        if progress_callback: progress_callback(0.3, "Cleaning and chunking text...")
        # 2. Clean Text
        cleaned_pages = TextCleaner.clean_document_pages(pages)
        
        # 3. Chunk Document
        chunk_size = settings.get("chunk_size", 1000)
        chunk_overlap = settings.get("chunk_overlap", 200)
        chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_document(cleaned_pages)
        
        if not chunks:
            raise ValueError(f"No text extracted or chunked from {filename}")

        if progress_callback: progress_callback(0.5, "Extracting case metadata...")
        # 4. Extract Document-level Metadata
        full_text = "\n".join([p["text"] for p in cleaned_pages])
        extractor = MetadataExtractor(
            ollama_url=settings.get("ollama_base_url", "http://localhost:11434"),
            ollama_model=settings.get("ollama_model", "qwen3:8b")
        )
        
        # If it is designated as a judgment, try to use LLM for metadata extraction
        doc_meta = extractor.extract(full_text, filename, use_llm=is_judgment)
        
        # Merge document-level metadata into chunk-level metadata
        for idx, chunk in enumerate(chunks):
            # Extract specific acts/sections mentioned in this chunk
            chunk_specific_meta = extractor.extract_chunk_metadata(chunk["text"])
            
            # Combine
            merged_meta = doc_meta.copy()
            # Merge lists uniquely
            merged_meta["acts"] = list(set(doc_meta.get("acts", [])).union(set(chunk_specific_meta.get("acts", []))))
            merged_meta["sections"] = list(set(doc_meta.get("sections", [])).union(set(chunk_specific_meta.get("sections", []))))
            merged_meta["page"] = chunk["metadata"].get("page", 1)
            merged_meta["filepath"] = str(filepath)
            
            chunk["metadata"] = merged_meta

        if progress_callback: progress_callback(0.6, "Saving document to SQLite...")
        # 5. Save to database
        db_doc_id = None
        cursor = self.conn.cursor()
        try:
            # Check if document already exists
            cursor.execute("SELECT id FROM documents WHERE filename = ?", (filename,))
            existing_row = cursor.fetchone()
            if existing_row:
                # If document exists, we delete it first to overwrite
                self.delete_document(existing_row[0])
                logger.info(f"Overwriting existing document '{filename}'")

            cursor.execute(
                """
                INSERT INTO documents (filename, filepath, file_type, file_size, chunk_count, is_judgment)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (filename, str(filepath), file_type, file_size, len(chunks), int(is_judgment))
            )
            db_doc_id = cursor.lastrowid
            
            # Save Chunks
            chunk_db_ids = []
            for idx, chunk in enumerate(chunks):
                cursor.execute(
                    """
                    INSERT INTO chunks (doc_id, chunk_index, text, metadata)
                    VALUES (?, ?, ?, ?)
                    """,
                    (db_doc_id, idx, chunk["text"], json.dumps(chunk["metadata"]))
                )
                chunk_db_ids.append(cursor.lastrowid)
                
            # If judgment, save details to judgments table
            if is_judgment:
                # Stringify lists
                acts_str = ", ".join(doc_meta.get("acts", []))
                secs_str = ", ".join(doc_meta.get("sections", []))
                kw_str = ", ".join(doc_meta.get("keywords", []))
                
                cursor.execute(
                    """
                    INSERT INTO judgments (
                        doc_id, case_name, court, bench, judge, judgment_date, citation, 
                        acts, sections, keywords, state, court_level, year, ratio_decidendi, important_paragraphs
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        db_doc_id,
                        doc_meta.get("case_name", "Unknown Case"),
                        doc_meta.get("court", "Unknown Court"),
                        doc_meta.get("bench", "Unknown Bench"),
                        doc_meta.get("judge", "Unknown Judge"),
                        doc_meta.get("judgment_date", "Unknown Date"),
                        doc_meta.get("citation", "Unknown Citation"),
                        acts_str,
                        secs_str,
                        kw_str,
                        doc_meta.get("state", "Unknown State"),
                        doc_meta.get("court_level", "Unknown Court Level"),
                        doc_meta.get("year"),
                        doc_meta.get("ratio_decidendi", ""),
                        doc_meta.get("important_paragraphs", "")
                    )
                )
            
            self.conn.commit()
            
        except Exception as db_err:
            self.conn.rollback()
            logger.error(f"Failed database insertion for {filename}: {db_err}")
            raise db_err

        if progress_callback: progress_callback(0.8, "Generating embeddings for FAISS...")
        # 6. Embed and Index in FAISS
        try:
            chunk_texts = [c["text"] for c in chunks]
            embeddings = self.embedding_generator.embed_documents(chunk_texts)
            self.faiss_manager.add_vectors(embeddings, chunk_db_ids)
        except Exception as faiss_err:
            logger.error(f"Error updating FAISS index for {filename}: {faiss_err}")
            # If FAISS indexing fails, roll back SQLite document addition to keep in sync
            self.delete_document(db_doc_id)
            raise faiss_err

        if progress_callback: progress_callback(0.9, "Rebuilding BM25 index...")
        # 7. Rebuild BM25 index with all chunks from database (to keep corpus intact)
        self.rebuild_bm25_index()
        
        if progress_callback: progress_callback(1.0, "Ingestion complete!")
        logger.info(f"Ingested {filename} successfully: {len(chunks)} chunks.")
        
        return {
            "doc_id": db_doc_id,
            "filename": filename,
            "chunks": len(chunks),
            "is_judgment": is_judgment
        }

    def delete_document(self, doc_id: int):
        """Deletes a document from SQLite, removes its vectors from FAISS, and rebuilds BM25."""
        logger.info(f"Deleting document ID: {doc_id}")
        
        # 1. Get Chunk IDs associated with the document
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))
        rows = cursor.fetchall()
        chunk_ids = [r[0] for r in rows]

        # 2. Delete vectors from FAISS index
        if chunk_ids and self.faiss_manager:
            self.faiss_manager.delete_vectors(chunk_ids)

        # 3. Delete from SQLite (foreign key cascade deletes chunks & judgments)
        try:
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            self.conn.commit()
            logger.info(f"Deleted document record {doc_id} from SQLite.")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error deleting document from SQLite: {e}")
            raise e

        # 4. Rebuild BM25 index
        self.rebuild_bm25_index()

    def rebuild_bm25_index(self):
        """Fetches all chunks from the database and rebuilds the BM25 index."""
        if not self.bm25_manager:
            return
            
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, text FROM chunks")
        rows = cursor.fetchall()
        
        chunks = [{"id": r[0], "text": r[1]} for r in rows]
        self.bm25_manager.build_index(chunks)

    def rebuild_all_indices(self, progress_callback=None):
        """Fully recreates both FAISS and BM25 indexes from existing database chunks."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, text FROM chunks")
        rows = cursor.fetchall()
        
        if not rows:
            logger.warning("No chunks found in database to rebuild index.")
            self.faiss_manager.clear()
            self.bm25_manager.clear()
            return
            
        total = len(rows)
        logger.info(f"Rebuilding all indices for {total} chunks.")
        
        # 1. Clear FAISS index
        self.faiss_manager.clear()
        
        # 2. Re-embed in batches to save memory
        batch_size = 64
        all_ids = []
        all_embeddings = []
        
        for i in range(0, total, batch_size):
            batch = rows[i:i+batch_size]
            batch_ids = [r[0] for r in batch]
            batch_texts = [r[1] for r in batch]
            
            if progress_callback:
                progress_callback(0.1 + (i / total) * 0.7, f"Embedding chunks {i+1} to {min(i+batch_size, total)} of {total}...")
                
            embeddings = self.embedding_generator.embed_documents(batch_texts)
            self.faiss_manager.add_vectors(embeddings, batch_ids)

        if progress_callback: progress_callback(0.85, "Rebuilding BM25 index...")
        # 3. Rebuild BM25
        chunks = [{"id": r[0], "text": r[1]} for r in rows]
        self.bm25_manager.build_index(chunks)
        
        if progress_callback: progress_callback(1.0, "All indices rebuilt successfully!")

    def get_statistics(self) -> Dict[str, Any]:
        """Gathers database statistics for the analytics dashboard."""
        stats = {
            "indexed_documents": 0,
            "judgments": 0,
            "total_chunks": 0,
            "acts_count": 0,
            "faiss_size": 0,
            "bm25_vocab_size": 0
        }
        
        try:
            cursor = self.conn.cursor()
            
            # Documents
            cursor.execute("SELECT COUNT(*) FROM documents")
            stats["indexed_documents"] = cursor.fetchone()[0]
            
            # Judgments
            cursor.execute("SELECT COUNT(*) FROM judgments")
            stats["judgments"] = cursor.fetchone()[0]
            
            # Chunks
            cursor.execute("SELECT COUNT(*) FROM chunks")
            stats["total_chunks"] = cursor.fetchone()[0]
            
            # Extract acts count from judgment table
            cursor.execute("SELECT acts FROM judgments")
            all_acts = set()
            for row in cursor.fetchall():
                if row[0]:
                    parts = [a.strip() for a in row[0].split(",")]
                    all_acts.update([p for p in parts if p])
            stats["acts_count"] = len(all_acts)
            
            # FAISS index size
            if self.faiss_manager and self.faiss_manager.index:
                stats["faiss_size"] = self.faiss_manager.index.ntotal
                
            # BM25 vocab size
            if self.bm25_manager and self.bm25_manager.bm25:
                stats["bm25_vocab_size"] = len(self.bm25_manager.bm25.doc_freqs)
                
        except Exception as e:
            logger.error(f"Error gathering database stats: {e}")
            
        return stats

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Returns all documents registered in the system."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, filename, filepath, file_type, file_size, uploaded_at, chunk_count, is_judgment FROM documents ORDER BY uploaded_at DESC")
        rows = cursor.fetchall()
        
        docs = []
        for r in rows:
            docs.append({
                "id": r[0],
                "filename": r[1],
                "filepath": r[2],
                "file_type": r[3],
                "file_size": r[4],
                "uploaded_at": r[5],
                "chunk_count": r[6],
                "is_judgment": bool(r[7])
            })
        return docs

    def close(self):
        """Closes SQLite database connection."""
        if self.conn:
            self.conn.close()
