import os
import sys
import logging
from pathlib import Path

# Ensure project root is in python path
sys.path.append(str(Path(__file__).resolve().parent))

from config import BASE_DIR
from app import get_rag_components

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LawGenerator")

# 1. Constitution of India reference content
CONSTITUTION_CONTENT = """
CONSTITUTION OF INDIA - BARE ACT REFERENCE

ARTICLE 14: EQUALITY BEFORE LAW
- Section: Article 14
- Act: Constitution of India
- Description: The State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India.
- Relevance: Protects citizens against arbitrary actions by police or state authorities. Ensures non-discriminatory treatment.

ARTICLE 19: PROTECTION OF CERTAIN RIGHTS REGARDING FREEDOM OF SPEECH, ASSEMBLY, ETC.
- Section: Article 19
- Act: Constitution of India
- Description: All citizens shall have the right to freedom of speech and expression, to assemble peaceably and without arms, to form associations or unions, to move freely throughout the territory of India, and to practice any profession, or to carry on any occupation, trade or business.

ARTICLE 21: PROTECTION OF LIFE AND PERSONAL LIBERTY
- Section: Article 21
- Act: Constitution of India
- Description: No person shall be deprived of his life or personal liberty except according to procedure established by law.
- Relevance: Crucial in cases of illegal police detention or wrongful confinement. Police must follow due procedure for arrest.
- Practical Next Steps: File a Habeas Corpus petition under Article 226 in the High Court or Article 32 in the Supreme Court if a person is illegally detained.
- Police Procedure: Arresting officer must prepare an arrest memo, inform a relative, and produce the accused before a Magistrate within 24 hours.

ARTICLE 22: PROTECTION AGAINST ARREST AND DETENTION IN CERTAIN CASES
- Section: Article 22
- Act: Constitution of India
- Description: No person who is arrested shall be detained in custody without being informed, as soon as may be, of the grounds for such arrest nor shall he be denied the right to consult, and to be defended by, a legal practitioner of his choice. Every person who is arrested and detained in custody shall be produced before the nearest magistrate within a period of twenty-four hours of such arrest.
- Police Procedure: Ground of arrest must be communicated. Mandatory production before Magistrate within 24 hours of arrest.
- Court Procedure: Apply for regular bail under Section 437/439 CrPC (or Section 480/482 BNSS) if produced before a Magistrate.
- Evidence Required: Arrest memo copy, station diary entries, medical examination report.
"""

# 2. BNS / IPC Criminal offenses reference content
CRIMINAL_LAW_CONTENT = """
BHARATIYA NYAYA SANHITA (BNS) & INDIAN PENAL CODE (IPC) - OFFENSES AND REMEDIES

THEFT (SECTION 303 BNS / SECTION 379 IPC)
- Section: Section 303 BNS (Formerly Section 379 IPC)
- Act: Bharatiya Nyaya Sanhita, 2023 / Indian Penal Code, 1860
- Description: Whoever, intending to take dishonestly any movable property out of the possession of any person without that person's consent, moves that property in order to such taking, is said to commit theft.
- Punishment: Imprisonment up to three years, or with fine, or both.
- Practical Next Steps for Stolen Phone/Property: Immediately block the IMEI number of the phone. Visit the nearest police station to file a First Information Report (FIR). If police refuse to file an FIR, send a written complaint to the Superintendent of Police (SP) under Section 154(3) CrPC (or Section 173(3) BNSS) or file a private complaint before the Magistrate.
- Evidence Required: Purchase bill of phone/property, IMEI number details, service provider confirmation, last tracked location screenshot.

CRIMINAL TRESPASS & ILLEGAL PROPERTY OCCUPATION (SECTION 329 BNS / SECTION 441 IPC)
- Section: Section 329 BNS (Formerly Section 441 IPC)
- Act: Bharatiya Nyaya Sanhita, 2023 / Indian Penal Code, 1860
- Description: Whoever enters into or upon property in the possession of another with intent to commit an offence or to intimidate, insult or annoy any person in possession of such property, or having lawfully entered, remains there unlawfully, commits criminal trespass.
- Relevance: Applicable to illegal property occupation or land grabbing.
- Practical Next Steps: File a police complaint for Criminal Trespass and Criminal Intimidation. File a civil suit under Section 6 of the Specific Relief Act, 1963, for quick recovery of possession of immovable property within 6 months of dispossession.
- Evidence Required: Property title deeds, possession certificates, utility bills showing prior possession, photographs/videos of the illegal occupation, police complaint receipt.

CRIMINAL INTIMIDATION & THREATS (SECTION 351 BNS / SECTION 503 & 506 IPC)
- Section: Section 351 BNS (Formerly Section 506 IPC)
- Act: Bharatiya Nyaya Sanhita, 2023 / Indian Penal Code, 1860
- Description: Whoever threatens another with any injury to his person, reputation or property, with intent to cause alarm to that person, or to cause that person to do any act which he is not legally bound to do, commits criminal intimidation.
- Practical Next Steps: File an FIR for criminal intimidation. If threats are via phone/social media, contact the Cyber Crime Cell.
- Evidence Required: Call recordings, screenshot of threat messages, CCTV footage, witness statements.

CHEATING & NON-PAYMENT OF SALARY (SECTION 318 BNS / SECTION 420 IPC)
- Section: Section 318 BNS (Formerly Section 420 IPC)
- Act: Bharatiya Nyaya Sanhita, 2023 / Indian Penal Code, 1860
- Description: Whoever cheats and thereby dishonestly induces the person deceived to deliver any property to any person, or to make, alter or destroy the whole or any part of a valuable security, commits cheating.
- Relevance: Applicable when an employer deliberately cheats employees by withholding salary with dishonest intention on day of hiring.
- Practical Next Steps: Send a legal notice to the employer demanding outstanding salary. File a claim under the Payment of Wages Act, 1936, or Industrial Disputes Act, 1947, before the Labor Commissioner. Alternatively, file an FIR for Cheating and Criminal Breach of Trust if dishonest intention is present from the start.
- Evidence Required: Employment contract, offer letter, salary slips, bank statements showing non-payment, communication logs (emails/chats with HR or boss), legal notice acknowledgment.

FALSE COMPLAINTS & DEFAMATION (SECTION 356 BNS / SECTION 499 & 500 IPC)
- Section: Section 356 BNS (Formerly Section 500 IPC)
- Act: Bharatiya Nyaya Sanhita, 2023 / Indian Penal Code, 1860
- Description: Whoever, by words spoken or intended to be read, makes or publishes any imputation concerning any person intending to harm the reputation of such person, commits defamation.
- Practical Next Steps (False complaint by wife or others): File an application under Section 482 of CrPC in the High Court for quashing the false FIR. Apply for Anticipatory Bail under Section 438 CrPC (Section 482 BNSS) to prevent arrest. File a counter-complaint for defamation or malicious prosecution under Section 211 IPC (Section 248 BNS).
- Evidence Required: Copy of false FIR, alibi proofs (flight tickets, office attendance logs proving you were elsewhere), call records, chats proving innocence or extortion motives.
"""

# 3. BNSS / CrPC Procedure reference content
PROCEDURE_LAW_CONTENT = """
BHARATIYA NAGARIK SURAKSHA SANHITA (BNSS) & CODE OF CRIMINAL PROCEDURE (CRPC) - PROCEDURES

POLICE PROCEDURE ON ARREST & DETENTION (SECTION 35 BNSS / SECTION 41A CRPC)
- Section: Section 35 BNSS (Formerly Section 41A CrPC)
- Act: Bharatiya Nagarik Suraksha Sanhita, 2023 / Code of Criminal Procedure, 1973
- Description: In all cases where the arrest of a person is not required under sub-section (1) of Section 35, the police officer shall issue a notice directing the person against whom a reasonable complaint has been made to appear before him.
- Relevance: Protects citizens from arbitrary arrest. Police cannot arrest without issuing a 41A notice first for offenses carrying less than 7 years imprisonment.
- Practical Next Steps: If you receive a Section 41A notice, you must comply and appear. If police threaten arrest despite compliance, file an Anticipatory Bail application.

FILING AN FIR (SECTION 173 BNSS / SECTION 154 CRPC)
- Section: Section 173 BNSS (Formerly Section 154 CrPC)
- Act: Bharatiya Nagarik Suraksha Sanhita, 2023 / Code of Criminal Procedure, 1973
- Description: Every information relating to the commission of a cognizable offence, if given orally to an officer in charge of a police station, shall be reduced to writing.
- Police Procedure: Police must register an FIR for cognizable offenses immediately. A copy of the FIR must be given to the informant free of cost.
- Practical Next Steps: If the local police refuse to register the FIR:
  1. Send the complaint in writing by registered post to the Superintendent of Police (SP) under Section 173(3) BNSS.
  2. If the SP fails to act, file a criminal complaint before the Magistrate under Section 156(3) CrPC / Section 175 BNSS requesting the court to direct the police to register an FIR and conduct investigation.
"""

def generate_and_index():
    # Write files to documents directory
    documents_dir = BASE_DIR / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_create = {
        "Constitution_of_India_Bare_Act.txt": CONSTITUTION_CONTENT,
        "BNS_IPC_Criminal_Law_Offenses.txt": CRIMINAL_LAW_CONTENT,
        "BNSS_CrPC_Police_Court_Procedure.txt": PROCEDURE_LAW_CONTENT
    }
    
    created_paths = []
    for name, content in files_to_create.items():
        filepath = documents_dir / name
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content.strip())
        logger.info(f"Created law reference file: {filepath.name}")
        created_paths.append(filepath)

    # Ingest the files
    logger.info("Initializing RAG database and starting ingestion...")
    doc_manager, _, _, _, _ = get_rag_components()
    
    for filepath in created_paths:
        logger.info(f"Ingesting: {filepath.name}")
        try:
            doc_manager.ingest_document(filepath, is_judgment=False)
            logger.info(f"Successfully ingested and indexed: {filepath.name}")
        except Exception as e:
            logger.error(f"Failed to ingest {filepath.name}: {e}")

    logger.info("Core law references successfully indexed! Rebuilding indexes...")
    doc_manager.rebuild_bm25_index()
    logger.info("All indexes updated. The Legal Advisor is now ready to handle queries!")

if __name__ == "__main__":
    generate_and_index()
