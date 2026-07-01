import json
import logging
import requests
from typing import List, Dict, Any, Generator

logger = logging.getLogger("LegalReasoner")

class LegalReasoner:
    """Interacts with the offline Ollama LLM, enforcing RAG constraints and structured output."""

    def __init__(self, ollama_url: str = "http://localhost:11434", model_name: str = "qwen3:8b"):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.disclaimer = (
            "\n\n---\n*Disclaimer: This application provides educational legal information "
            "generated from locally indexed legal materials. It is not legal advice and should "
            "not replace consultation with a qualified advocate.*"
        )

    def _build_prompt(self, query: str, chunks: List[Dict[str, Any]], language: str = "English", chat_history: List[List[str]] = None) -> str:
        """Constructs a prompt enforcing strict context adherence and output format."""
        
        # Format the context
        context_str = ""
        for idx, chunk in enumerate(chunks):
            meta = chunk.get("metadata", {})
            doc_name = meta.get("filename", "Unknown Document")
            page = meta.get("page", "?")
            context_str += f"--- CONTEXT CHUNK {idx+1} (Source: {doc_name}, Page: {page}) ---\n"
            context_str += chunk.get("text", "") + "\n\n"

        # Format conversation history from list of dictionaries
        history_str = ""
        if chat_history:
            history_str = "CONVERSATION HISTORY:\n"
            for msg in chat_history:
                # msg is a dict, e.g. {"role": "user", "content": "..."}
                if isinstance(msg, dict):
                    role = msg.get("role", "user").capitalize()
                    content = msg.get("content", "")
                    if role and content:
                        history_str += f"{role}: {content}\n\n"
                    
        history_block = f"\n{history_str}\n" if history_str else ""

        lang_instruction = ""
        if language.lower() == "hindi":
            lang_instruction = (
                "You must write the legal advisory response in HINDI. Keep the section headers in Hindi, "
                "but you can write specific legal terms and citations in both Hindi and English for clarity."
            )
        else:
            lang_instruction = "You must write the response in ENGLISH."

        prompt = f"""You are Nyaya Mitra, an expert conversational Indian Legal AI Advisor. Your goal is to guide the user through their legal situation interactively, gather facts, and provide structured legal advisory using the retrieved local context.

RETRIVED LEGAL CONTEXT (from local law databases):
{context_str}
{history_block}
USER CURRENT MESSAGE:
"{query}"

INSTRUCTIONS:
1. If the user is starting a conversation or if the situation description is brief or lacks essential details, do NOT give the final legal advice yet. Instead, ask the user polite, clarifying questions (one at a time) to gather necessary facts (e.g., "What exactly is the dispute?", "Do you have a written contract?", "Has an FIR been filed?", "When did this happen?").
2. Keep the conversation engaging, clear, and empathetic. 
3. If you have gathered enough details, or if the user asks you to provide your legal assessment/remessage directly, use the RETRIEVED LEGAL CONTEXT to generate a comprehensive legal advisory.
4. When delivering the final legal advisory, you MUST structure your response exactly using these headings:

### Legal Assessment: Who is Right and Who is Wrong
[An objective assessment of which party's legal position is stronger under the retrieved laws, who violated procedures, and who is in the right.]

### Actionable Remediations (Who must do What)
[Specific actions that need to be taken by the user/client and by the opposite party (or police/authorities) to resolve the situation.]

### Opposition's Potential Counters and Defenses
[List potential arguments, objections, or defenses that the opposing lawyer could raise to challenge your position (e.g., lack of evidence, delay/limitations, procedural defaults), and what you must take care of to counter them.]

### Recommended Legal Steps & Evidence Required
[Practical, step-by-step legal steps to take (filing complaints, sending notices) and the exact evidence (receipts, contract copies, chat logs) needed to support them.]

### Applicable Acts & Sections
[Citations of the relevant Acts, Sections, and Constitutional Articles found in the retrieved context.]

5. If the retrieved context does not contain sufficient details to address the user's issue, clearly state: "Insufficient supporting legal material was found in the indexed database to answer this question." Do not invent any laws, case names, or citations.
6. {lang_instruction}
"""
        return prompt

    def generate_response_stream(
        self, 
        query: str, 
        chunks: List[Dict[str, Any]], 
        language: str = "English",
        temperature: float = 0.2,
        context_length: int = 8192,
        chat_history: List[List[str]] = None
    ) -> Generator[str, None, None]:
        """Queries Ollama and streams the response chunk by chunk."""
        prompt = self._build_prompt(query, chunks, language, chat_history)
        
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_ctx": context_length
            }
        }
        
        logger.info(f"Sending prompt to Ollama model '{self.model_name}' (Length: {len(prompt)} chars)")
        
        try:
            # Send streaming POST request
            response = requests.post(url, json=payload, stream=True, timeout=120)
            
            if response.status_code != 200:
                error_msg = f"Ollama returned HTTP {response.status_code}: {response.text}"
                logger.error(error_msg)
                yield f"Error connecting to Ollama. {error_msg}"
                yield self.disclaimer
                return
                
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8'))
                        chunk_text = chunk.get("response", "")
                        full_response += chunk_text
                        yield full_response
                    except Exception as parse_err:
                        logger.error(f"Error parsing Ollama stream chunk: {parse_err}")
                        
            # Yield final disclaimer at the very end
            yield full_response + self.disclaimer
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama connection error: {e}")
            yield f"Could not connect to Ollama. Please check if the Ollama service is running on {self.ollama_url} and the model '{self.model_name}' is installed."
            yield self.disclaimer
