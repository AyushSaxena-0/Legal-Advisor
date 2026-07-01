import os
import argparse
import logging
from pathlib import Path
import sys

# Ensure project root is in python path
sys.path.append(str(Path(__file__).resolve().parent))

from config import BASE_DIR, load_settings
from app import get_rag_components

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("BulkIngester")

def bulk_ingest(years: list = None, limit: int = None):
    judgments_dir = BASE_DIR / "supreme_court_judgments"
    if not judgments_dir.exists():
        logger.error(f"Supreme Court Judgments folder not found at: {judgments_dir}")
        return

    logger.info("Initializing RAG database and models...")
    doc_manager, _, _, _, _ = get_rag_components()

    # Collect all PDF files in selected year folders
    pdf_files = []
    
    # Sort folders to process chronologically
    sorted_items = sorted(os.listdir(judgments_dir))
    
    for item in sorted_items:
        item_path = judgments_dir / item
        if item_path.is_dir() and item.isdigit():
            # Check if this year matches filter
            if years and item not in years:
                continue
                
            logger.info(f"Scanning year folder: {item}")
            for filename in os.listdir(item_path):
                if filename.lower().endswith(".pdf"):
                    pdf_files.append(item_path / filename)

    total_found = len(pdf_files)
    logger.info(f"Found total of {total_found} PDF judgments to index.")
    
    if limit:
        pdf_files = pdf_files[:limit]
        logger.info(f"Applying limit: Ingesting first {len(pdf_files)} files.")

    if not pdf_files:
        logger.info("No files selected for ingestion.")
        return

    success = 0
    failed = 0

    for idx, filepath in enumerate(pdf_files):
        logger.info(f"[{idx+1}/{len(pdf_files)}] Processing: {filepath.name} (Year: {filepath.parent.name})")
        try:
            doc_manager.ingest_document(filepath, is_judgment=True)
            success += 1
        except Exception as e:
            logger.error(f"Failed to ingest {filepath.name}: {e}")
            failed += 1

    logger.info("========================================")
    logger.info(f"Bulk Ingestion Summary:")
    logger.info(f"Processed: {len(pdf_files)}")
    logger.info(f"Success: {success}")
    logger.info(f"Failed: {failed}")
    logger.info("========================================")
    logger.info("If you added new items, remember to check the database stats in the app.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk Ingest Supreme Court Judgments into Legal RAG")
    parser.add_argument("--years", type=str, help="Comma-separated list of years to process (e.g. 1950,1951)")
    parser.add_argument("--limit", type=int, help="Limit total number of documents to ingest (for testing)")
    
    args = parser.parse_args()
    
    year_list = [y.strip() for y in args.years.split(",")] if args.years else None
    
    bulk_ingest(years=year_list, limit=args.limit)
