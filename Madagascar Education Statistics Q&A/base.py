#============================Imports============================#
from dotenv import load_dotenv, find_dotenv
from typing import List, Tuple, Dict
from pypdf import PdfReader
import pdfplumber
from tqdm import tqdm
import google.generativeai as genai
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import pandas as pd
import csv
import re
import os
import warnings
import time
import google.api_core.exceptions

#============================Config============================#
warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv(find_dotenv())
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    os.environ["GEMINI_API_KEY"] = api_key
print(api_key)

#============================Load PDF============================#
def load_pdf(file_path: str) -> Tuple[List[str], List[Dict]]:
    reader = PdfReader(file_path)
    docs = []
    meta = []
    for num, page in enumerate(tqdm(reader.pages, desc=f"Lire {os.path.basename(file_path)}"), start=1):
        text = page.extract_text()
        if text and text.strip():
            docs.append(text.strip())
            meta.append({
                "filename": os.path.basename(file_path),
                "page_number": num
            })
    print("pdf chargé..")
    return docs, meta

#============================Extract Tables============================#
def extract_tables(file: str) -> List[Dict]:
    tbls = []
    with pdfplumber.open(file) as pdf:
        for pg, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for tbl in tables:
                tbls.append({
                    "table": tbl,
                    "page_num": pg
                })
    print("tableaux extraits..")
    return tbls

#============================Table to Text============================#
def table_to_text(tbl: List[List[str]]) -> str:
    return "\n".join(["\t".join(str(cell or "") for cell in row) for row in tbl if row])

#============================Load Data============================#
def load_data(documents: List[str], metadatas: List[Dict], collection_name: str, tables: List[Dict]):
    client = chromadb.EphemeralClient()
    embed = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    coll = client.create_collection(
        name=collection_name, embedding_function=embed
    )
    for tbl in tables:
        tbl_text = table_to_text(tbl["table"])
        documents.append(tbl_text)
        metadatas.append({
            "filename": "tableau",
            "page_number": tbl["page_num"]
        })
    coll.add(
        documents=documents,
        metadatas=metadatas,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
    print("données chargées...")
    return coll

#============================Get Passage============================#
def get_relevant_passage(query: str, db, n_results=3):
    res = db.query(
        query_texts=[query], n_results=n_results, include=["documents", "metadatas"]
    )
    ctx = " ".join(res['documents'][0])
    meta = res['metadatas'][0][0]
    return ctx, meta

#============================Classify Question============================#
def classify_q(q: str) -> str:
    q = q.lower()
    if "nombre" in q or "effectif" in q:
        return "num"
    elif "pourcentage" in q or "%" in q:
        return "pct"
    elif "taux de réussite" in q:
        return "success"
    return "gen"

#============================Extract Number============================#
def extract_num(ans: str) -> str:
    match = re.search(r'\b\d+\b|[\d.]+%|[\d,]+\b', ans)
    return match.group(0) if match else ans

#============================RAG Prompt============================#
def make_rag_prompt(query: str, context: str) -> str:
    return f"""
    Expert en stats éducatives Madagascar. Répondez en français, basé sur le contexte. Soyez précis, concis, factuel. Pas d'info hors contexte. Pour nombres/pourcentages, donnez exactement comme dans le contexte.

    **Question**: {query}
    **Contexte**: {context}
    **Réponse**:
    """

#============================Gemini Response============================#
def get_gemini_response(query: str, context: str) -> str:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    q_type = classify_q(query)
    prompt = make_rag_prompt(query, context)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    max_retries = 3
    retry_delay = 32  # Initial delay based on API suggestion
    for attempt in range(max_retries):
        try:
            res = model.generate_content(prompt)
            time.sleep(4)  # Ensure < 15 RPM (60s / 15 = 4s per request)
            return extract_num(res.text) if q_type in ["num", "pct", "success"] else res.text
        except google.api_core.exceptions.ResourceExhausted as e:
            if attempt < max_retries - 1:
                print(f"Quota dépassé pour la question '{query}'. Réessai dans {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise e
    return "Erreur : Quota dépassé après plusieurs tentatives"

#============================Process CSV============================#
def process_questions_from_csv(db, csv_path: str, output_csv: str = 'submission_file.csv'):
    df = pd.read_csv(csv_path)
    # Write header if file doesn't exist
    if not os.path.exists(output_csv):
        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "question", "answer", "context", "ref_page"])
            writer.writeheader()
    
    for _, row in df.iterrows():
        qid = row["id"]
        question = row["question"]
        print(f"Traitement de la question ID {qid}: {question}")
        txt, meta = get_relevant_passage(question, db, n_results=3)
        if not txt:
            ans = "Info non trouvée"
            ctx = "Aucun contexte extrait"
            pg = "N/A"
        else:
            ans = get_gemini_response(question, txt)
            ctx = txt
            pg = meta["page_number"]
        print(f"Réponse générée: {ans}")
        # Append result to CSV immediately
        with open(output_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "question", "answer", "context", "ref_page"])
            writer.writerow({
                "id": qid,
                "question": question,
                "answer": ans,
                "context": str(ctx),
                "ref_page": str(pg)
            })
        print(f"Résultat écrit dans {output_csv} pour ID {qid}")

#============================Main============================#
data, metadata = load_pdf(file_path="./data/MESUPRES_en_chiffres_MAJ.pdf")
tbls = extract_tables(file="./data/MESUPRES_en_chiffres_MAJ.pdf")
coll_name = 'rag'
db = load_data(documents=data, metadatas=metadata, collection_name=coll_name, tables=tbls)
process_questions_from_csv(db, './data/questions.csv')

#============================Main============================#