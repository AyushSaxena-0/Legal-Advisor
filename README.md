# ⚖️ Nyaya Mitra: Offline Retrieval-Augmented Indian Legal Intelligence System

Nyaya Mitra is a production-ready, completely **OFFLINE** Indian AI Legal Advisor that runs on Windows. It combines local Retrieval-Augmented Generation (RAG) using FAISS and BM25 hybrid retrieval, MiniLM Cross-Encoder re-ranking, and the Qwen3:8B model running through local Ollama inference. 

<img width="1890" height="862" alt="image" src="https://github.com/user-attachments/assets/0d944465-9d6c-4f53-8b27-9b191f188478" />



<img width="1885" height="861" alt="image" src="https://github.com/user-attachments/assets/80abce4f-d656-4b24-b36d-f06167151c3c" />



<img width="1885" height="867" alt="image" src="https://github.com/user-attachments/assets/b8abe322-8ba6-41a4-9897-7a82c455a56c" />



<img width="1888" height="850" alt="image" src="https://github.com/user-attachments/assets/e9a90155-150a-48b1-8125-71cc30eb4e76" />


---

## 🚀 Key Features

* **Complete Offline Security**: Zero API keys or cloud transmissions. All vectors, database entries, and LLM text generation occur locally on your machine.
* **Hybrid Search (FAISS + BM25)**: Blends semantic dense retrieval (`0.70`) with lexical sparse search (`0.30`) to fetch context chunks accurately in both English and Hindi.
* **MiniLM Reranking**: Utilizes `cross-encoder/ms-marco-MiniLM-L-6-v2` to select the top 8 most contextually relevant legal paragraphs from 20 hybrid candidates.
* **Robust Ingestion**: Supports `.pdf` (text & scanned OCR), `.docx` (with zip/xml fallbacks), `.txt`, `.csv`, `.md`, and `.json`.
* **Citations & Guardrails**: Analyzes LLM output using a rule-based `CitationValidator` against source chunks to highlight unverified citations and sections.
* **Dual-Language Capabilities**: Analyzes user situations and provides advisory outputs in English or Hindi.
* **Modern Gradio UI**: Sleek, customizable dark modern dashboard with tabs for Consulting, Search, Explorer, Ingestion, and Analytics.

---

## 🛠️ Tech Stack

* **Language**: Python 3.11+
* **Vector Store**: FAISS
* **Keyword Index**: Rank-BM25
* **Embeddings**: Sentence-Transformers (`BAAI/bge-base-en-v1.5` or `intfloat/e5-large-v2`)
* **Local LLM**: Ollama (`qwen3:8b`)
* **Document Extraction**: PyMuPDF (fitz) & RapidOCR (ONNX runtime for scanned PDFs)
* **Metadata & Logs**: SQLite & standard python logging

---

## 📂 Database & Data Sources

This RAG advisor retrieves legal precedents and raw bare acts from local indexed documents. To populate your knowledge base, we leverage:
* **Constitution of India (COI)**: The official, up-to-date PDF of the Constitution of India.
  * **Source Link**: [Ministry of Law and Justice - Constitution of India PDF](https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2024/07/20240716890312078.pdf)
  * **Usage**: Save the downloaded file as `COI.pdf` in the project root directory and run `python ingest_coi.py` to index the entire Constitution of India.
* **Legal Dataset (SC Judgments India 1950-2024)**: Over 26,600 Supreme Court judgments in PDF format.
  * **Source Link**: [Kaggle - Legal Dataset SC Judgments India (1950-2024)](https://www.kaggle.com/datasets/adarshsingh0903/legal-dataset-sc-judgments-india-19502024)
  * **Usage**: Download and place the year-wise subdirectories (e.g., `1950/`, `1951/`, etc.) directly inside the `supreme_court_judgments/` directory. Use the `ingest_judgments.py` script to index them into your local databases.

---

## 📋 System Requirements & Installation

### 1. Prerequisites
* **Python**: Version 3.11 or higher installed on Windows.
* **Ollama**: Download and install Ollama for Windows from [ollama.com](https://ollama.com).
* **GPU (Recommended)**: NVIDIA GPU with CUDA capability (8GB+ VRAM) for accelerated inference.

### 2. Pull Ollama Model
Start Ollama and pull the target Qwen model:
```cmd
ollama pull qwen2.5:7b
```
*(Note: If using a specific `qwen3:8b` tags variant, replace the pull tag. You can configure your target model tag directly in the Settings tab of the application).*

### 3. Setup Virtual Environment
Clone or navigate to the directory `D:\Legal\legal_ai` and install the package requirements:
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## ⚡ Running the Application

To launch the local web server:
```cmd
python app.py
```
After initialization completes, open your web browser and navigate to:
**[http://127.0.0.1:7860](http://127.0.0.1:7860)**

---

## 📂 Project Structure

```text
legal_ai/
├── app.py                  # Main Gradio application code
├── config.py               # Paths, logger & configuration settings saver
├── requirements.txt        # Package dependencies
├── README.md               # Setup & project documentation
├── database/               # SQLite database & configuration settings
├── documents/              # Cached user uploaded reference files
├── embeddings/             # HF Embedding models cache
├── faiss_index/            # Persistent FAISS indices
├── bm25_index/             # Pickled BM25 indexes
├── models/                 # Cached local Hugging Face model files
├── rag/                    # RAG modules
│   ├── __init__.py
│   ├── pdf_loader.py
│   ├── ocr_loader.py
│   ├── text_cleaner.py
│   ├── chunker.py
│   ├── metadata_extractor.py
│   ├── embedding_generator.py
│   ├── faiss_manager.py
│   ├── bm25_manager.py
│   ├── hybrid_retriever.py
│   ├── cross_encoder_reranker.py
│   ├── case_retriever.py
│   ├── legal_reasoner.py
│   ├── response_formatter.py
│   └── citation_validator.py
└── logs/                   # System runtime logs
```

---

## ⚖️ Disclaimer & Warnings

> [!WARNING]
> **Educational Use Only**: This application provides educational legal information generated from locally indexed legal materials. It is not legal advice and should not replace consultation with a qualified, registered advocate.
> 
> **Database Dependency**: RAG accuracy is strictly limited by the documents you choose to ingest. If a law, section, or precedent is missing from your local database directories, the assistant will output "Insufficient supporting legal material". Always verify citations and sections against original official government gazettes.
> 
> **Hardware & Memory Limitations**: Ingesting the complete 26,000+ court judgments dataset requires substantial CPU/GPU processing power and time. To prevent local hardware freezes or Out Of Memory (OOM) crashes, run bulk ingestion in batches using the `--limit` or `--years` flags.
