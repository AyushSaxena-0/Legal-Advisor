import os
import sys
import shutil
from pathlib import Path

# Ensure project root is in python path
sys.path.append(str(Path(__file__).resolve().parent))

from config import BASE_DIR, DOCUMENTS_DIR
from app import get_rag_components

def ingest_coi():
    coi_src = BASE_DIR / "COI.pdf"
    coi_dest = DOCUMENTS_DIR / "COI.pdf"
    
    if not coi_src.exists():
        print(f"Source COI.pdf not found at: {coi_src}")
        return
        
    print(f"Copying COI.pdf to: {coi_dest}")
    shutil.copy2(coi_src, coi_dest)
    
    print("Initializing RAG database and models...")
    doc_manager, _, _, _, _ = get_rag_components()
    
    print("Ingesting COI.pdf...")
    try:
        res = doc_manager.ingest_document(coi_dest, is_judgment=False)
        print(f"Successfully ingested COI.pdf: {res}")
    except Exception as e:
        print(f"Error during ingestion: {e}")

if __name__ == "__main__":
    ingest_coi()
