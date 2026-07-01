import re
import json
import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger("MetadataExtractor")

class MetadataExtractor:
    """Extracts document-level and chunk-level metadata using regex and optional LLM calls."""

    # Regex patterns for chunk-level extraction
    SECTION_PATTERN = re.compile(
        r'(?:section|sec\.|u/s|u/ss)\s*([\d\w]+(?:\s*(?:to|,|and)\s*[\d\w]+)*)', 
        re.IGNORECASE
    )
    
    # Common Indian Acts pattern
    ACT_PATTERNS = [
        re.compile(r'(indian\s+penal\s+code|ipc)', re.IGNORECASE),
        re.compile(r'(code\s+of\s+criminal\s+procedure|crpc)', re.IGNORECASE),
        re.compile(r'(indian\s+evidence\s+act|iea)', re.IGNORECASE),
        re.compile(r'(bharatiya\s+nyaya\s+sanhita|bns)', re.IGNORECASE),
        re.compile(r'(bharatiya\s+nagarik\s+suraksha\s+sanhita|bnss)', re.IGNORECASE),
        re.compile(r'(bharatiya\s+sakshya\s+adhiniyam|bsa)', re.IGNORECASE),
        re.compile(r'(constitution\s+of\s+india|constitution)', re.IGNORECASE),
        re.compile(r'(civil\s+procedure\s+code|cpc)', re.IGNORECASE),
        re.compile(r'([A-Za-z\s]+act,\s+\d{4})', re.IGNORECASE),
    ]

    COURT_LEVEL_PATTERNS = {
        "Supreme Court": re.compile(r'(supreme\s+court\s+of\s+india|insc|s\.c\.)', re.IGNORECASE),
        "High Court": re.compile(r'(high\s+court|h\.c\.)', re.IGNORECASE),
        "District Court": re.compile(r'(district\s+court|sessions\s+court|magistrate)', re.IGNORECASE)
    }

    INDIAN_STATES = [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
        "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
        "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan",
        "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
        "Delhi", "Jammu and Kashmir", "Ladakh", "Puducherry"
    ]

    def __init__(self, ollama_url: str = "http://localhost:11434", ollama_model: str = "qwen3:8b"):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    def extract_chunk_metadata(self, text: str) -> Dict[str, Any]:
        """Extracts mentions of Acts and Sections directly from the chunk text using regex."""
        metadata = {}
        
        # Extract Acts
        acts = set()
        for pattern in self.ACT_PATTERNS:
            matches = pattern.findall(text)
            for m in matches:
                if isinstance(m, tuple):
                    m = m[0]
                # Normalize common ones
                m_clean = m.strip().upper()
                if "INDIAN PENAL CODE" in m_clean or m_clean == "IPC":
                    acts.add("IPC (Indian Penal Code)")
                elif "CRIMINAL PROCEDURE" in m_clean or m_clean == "CRPC":
                    acts.add("CrPC (Code of Criminal Procedure)")
                elif "EVIDENCE ACT" in m_clean or m_clean == "IEA":
                    acts.add("Indian Evidence Act")
                elif "BHARATIYA NYAYA SANHITA" in m_clean or m_clean == "BNS":
                    acts.add("BNS (Bharatiya Nyaya Sanhita)")
                elif "BHARATIYA NAGARIK" in m_clean or m_clean == "BNSS":
                    acts.add("BNSS (Bharatiya Nagarik Suraksha Sanhita)")
                elif "BHARATIYA SAKSHYA" in m_clean or m_clean == "BSA":
                    acts.add("BSA (Bharatiya Sakshya Adhiniyam)")
                elif "CONSTITUTION" in m_clean:
                    acts.add("Constitution of India")
                elif "CIVIL PROCEDURE" in m_clean or m_clean == "CPC":
                    acts.add("CPC (Code of Civil Procedure)")
                else:
                    acts.add(m.strip())
        
        metadata["acts"] = list(acts)

        # Extract Sections
        sections = set()
        sections_found = self.SECTION_PATTERN.findall(text)
        for s in sections_found:
            # Clean section string (e.g. split by commas or "to")
            s_clean = s.strip()
            # If multiple sections like "302 and 304", split them
            split_secs = re.split(r'\s*(?:,|and|to)\s*', s_clean)
            for s_sub in split_secs:
                if s_sub.isalnum():
                    sections.add(s_sub)
        
        metadata["sections"] = list(sections)
        return metadata

    def extract_document_metadata_rules(self, text: str, filename: str) -> Dict[str, Any]:
        """Rule-based heuristic extractor as a fallback for document-level metadata."""
        meta = {
            "document_name": filename,
            "case_name": "Unknown Case",
            "court": "Unknown Court",
            "judge": "Unknown Judge",
            "bench": "Unknown Bench",
            "judgment_date": "Unknown Date",
            "citation": "Unknown Citation",
            "acts": [],
            "sections": [],
            "keywords": [],
            "state": "Unknown State",
            "court_level": "Unknown Court Level",
            "year": None
        }

        # Clean filename to set as default case name
        case_name_guess = Path(filename).stem.replace("_", " ").replace("-", " ")
        meta["case_name"] = case_name_guess

        # Guess court level
        for level, pattern in self.COURT_LEVEL_PATTERNS.items():
            if pattern.search(text[:2000]):
                meta["court_level"] = level
                if level == "Supreme Court":
                    meta["court"] = "Supreme Court of India"
                break
        
        # Try to find a year
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', text[:2000])
        if year_match:
            meta["year"] = int(year_match.group(1))

        # Try to find a date
        date_pattern = re.compile(
            r'(\b\d{1,2}(?:st|nd|rd|th)?[-/\s]+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-/\s]+\d{2,4})|'
            r'(\b\d{1,2}[-/\s]+\d{1,2}[-/\s]+\d{2,4})', 
            re.IGNORECASE
        )
        date_match = date_pattern.search(text[:2000])
        if date_match:
            meta["judgment_date"] = date_match.group(0).strip()
            # Try to extract year from date
            yr_match = re.search(r'\b(19\d{2}|20\d{2})\b', meta["judgment_date"])
            if yr_match:
                meta["year"] = int(yr_match.group(1))

        # Guess State
        for state in self.INDIAN_STATES:
            if re.search(r'\b' + re.escape(state) + r'\b', text[:2000], re.IGNORECASE):
                meta["state"] = state
                break

        # Extracts Acts and Sections using regex
        chunk_meta = self.extract_chunk_metadata(text[:4000])
        meta["acts"] = chunk_meta["acts"]
        meta["sections"] = chunk_meta["sections"]

        return meta

    def extract_document_metadata_llm(self, text: str, filename: str) -> Dict[str, Any]:
        """Queries local Ollama to extract metadata from document header."""
        # Use first 3000 chars which contains title page info
        header_text = text[:3000]
        
        prompt = f"""
You are a highly precise legal metadata extraction system. Analyze the following legal text from the beginning of an Indian court document and extract the metadata in JSON format.

Text:
\"\"\"
{header_text}
\"\"\"

Extract the following keys exactly:
- case_name (Name of the case, e.g., "State of Maharashtra v. XYZ" or "K.S. Puttaswamy v. Union of India")
- court (Name of the court, e.g., "Supreme Court of India", "High Court of Delhi")
- judge (Names of the judge(s) who delivered the judgment)
- bench (Bench description, e.g., "Division Bench", "Constitutional Bench", "Single Judge")
- judgment_date (Date of judgment, e.g., "YYYY-MM-DD" or standard date string)
- citation (Court citation, e.g., "2023 INSC 123", "AIR 2021 SC 432")
- acts (List of Acts mentioned, e.g., ["Indian Penal Code", "Code of Criminal Procedure"])
- sections (List of specific sections mentioned, e.g., ["302", "307", "482"])
- keywords (List of 3-5 legal keywords, e.g., ["Bail", "Murder", "Quashing"])
- state (Indian State where court is located)
- court_level ("Supreme Court", "High Court", or "District Court")
- year (Four-digit year of the judgment)

Your output must be ONLY a valid JSON object. Do not include any explanation or extra text.
"""

        try:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_ctx": 4096
                }
            }
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=30)
            if response.status_code == 200:
                resp_json = response.json()
                raw_response = resp_json.get("response", "").strip()
                
                # Try to find JSON block in output
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    extracted_meta = json.loads(json_match.group(0))
                    # Basic checks and defaults
                    extracted_meta["document_name"] = filename
                    
                    # Ensure lists are lists
                    for list_key in ["acts", "sections", "keywords"]:
                        if list_key in extracted_meta and not isinstance(extracted_meta[list_key], list):
                            extracted_meta[list_key] = [extracted_meta[list_key]]
                            
                    return extracted_meta
            
            logger.warning("Ollama response was not successful or did not contain valid JSON metadata. Falling back to rules.")
        except Exception as e:
            logger.error(f"Error calling local Ollama for metadata: {e}. Falling back to rules.")

        # Fallback to rules if LLM fails
        return self.extract_document_metadata_rules(text, filename)

    def extract(self, text: str, filename: str, use_llm: bool = True) -> Dict[str, Any]:
        """Facade method to extract document-level metadata."""
        if use_llm:
            return self.extract_document_metadata_llm(text, filename)
        else:
            return self.extract_document_metadata_rules(text, filename)
