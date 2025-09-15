#============================Imports============================#
from dotenv import load_dotenv, find_dotenv
from typing import List, Tuple, Dict
import pdfplumber
from tqdm import tqdm
import google.generativeai as genai
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import pandas as pd
import csv
import re, logging
import difflib
import os
import warnings
import time
import google.api_core.exceptions

#============================Config============================#
logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv(find_dotenv())
api_keys = [
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_API_KEY_SECOND"),
    os.getenv("GEMINI_API_KEY_THIRD"),
    os.getenv("GEMINI_API_KEY_FOURTH"),
]
api_keys = [key for key in api_keys if key]
if not api_keys:
    raise ValueError("No valid Gemini API keys found in environment variables")

#============================Load PDF============================#
def load_pdf(file_path: str) -> Tuple[List[str], List[Dict]]:
    docs = []
    meta = []
    with pdfplumber.open(file_path) as pdf:
        for num, page in enumerate(tqdm(pdf.pages, desc=f"Lecture de {os.path.basename(file_path)}"), start=1):
            text = page.extract_text() or ""
            if text:
                # Clean text
                text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\{\}\'\"\«\»]', '', text)
                text = re.sub(r'\s+', ' ', text).strip().replace('\n', ' ').replace('\r', ' ')
                docs.append(text)
                meta.append({
                    "filename": os.path.basename(file_path),
                    "page_number": num
                })
    return docs, meta

#============================Extract Tables============================#
def extract_tables(file: str) -> List[Dict]:
    tbls = []
    with pdfplumber.open(file) as pdf:
        for pg, page in enumerate(tqdm(pdf.pages, desc="Extraction des tableaux"), start=1):
            tables = page.extract_tables()
            for tbl in tables:
                tbls.append({
                    "table": tbl,
                    "page_num": pg
                })
    return tbls

#============================Table to Text============================#
def table_to_text(tbl: List[List[str]]) -> str:
    formatted = "Tableau:\n"
    for row in tbl:
        formatted += " ".join(str(cell or "") for cell in row) + "\n"
    return formatted.replace('\n', ' ').replace('\r', ' ')

#============================Load Data============================#
def load_data(documents: List[str], metadatas: List[Dict], collection_name: str, tables: List[Dict]):
    client = chromadb.EphemeralClient()
    embed = SentenceTransformerEmbeddingFunction(model_name="distiluse-base-multilingual-cased-v1")
    coll = client.create_collection(
        name=collection_name, embedding_function=embed
    )
    for tbl in tqdm(tables, desc="Ajout des tableaux à la collection"):
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
    return coll

#============================Get Passage============================#
def tokenize(text: str):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text.split()

def similarity_score(query: str, text: str) -> float:
    q_tokens = tokenize(query)
    t_tokens = tokenize(text)
    seq_matcher = difflib.SequenceMatcher(None, q_tokens, t_tokens)
    difflib_score = seq_matcher.ratio()
    match = seq_matcher.find_longest_match(0, len(q_tokens), 0, len(t_tokens))
    lcs = match.size if match.size > 0 else 0
    common_words = len(set(q_tokens) & set(t_tokens)) / max(1, len(q_tokens))
    final_score = (0.5 * difflib_score) + (0.3 * common_words) + (0.2 * (lcs / max(1, len(q_tokens))))
    return final_score

def get_relevant_passage(query: str, db, n_results=10):
    print("Getting relevant passage...")
    res = db.query(query_texts=[query], n_results=n_results, include=["documents", "metadatas"])
    print(res)
    candidates = res['documents'][0]
    metas = res['metadatas'][0]

    scored = [(similarity_score(query, doc), doc, meta) for doc, meta in zip(candidates, metas)]
    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best_doc, best_meta = scored[0]
    return best_doc, best_meta

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
    Expert en stats éducatives Madagascar. Répondez en français, basé sur le contexte. Soyez précis, concis, factuel. Pas d'info hors contexte. Pour nombres/pourcentages, donnez exactement comme dans le contexte. Pour réponses textuelles, donnez uniquement la réponse attendue sans explication, sans calcul, sans phrase supplémentaire. Réponse directe seulement.

    **Question**: {query}
    **Contexte**: {context}
    **Réponse**:
    """

#============================Gemini Response============================#
def get_gemini_response(query: str, context: str) -> str:
    q_type = classify_q(query)
    prompt = make_rag_prompt(query, context)
    model = genai.GenerativeModel("gemini-2.5-flash")
    max_retries = 3
    retry_delay = 5
    for key_index, api_key in enumerate(api_keys):
        try:
            genai.configure(api_key=api_key)
            for attempt in range(max_retries):
                try:
                    res = model.generate_content(prompt)
                    time.sleep(0.5)
                    res = res.text.strip().replace('\n', ' ').replace('\r', ' ')
                    res = extract_num(res) if q_type in ["num", "pct", "success"] else res
                    return res + "%" if q_type == "pct" else res
                except google.api_core.exceptions.ResourceExhausted as e:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 1.1
                    else:
                        break
        except Exception:
            continue
    return "Erreur : Toutes les clés API ont dépassé leur quota ou ont échoué"

#============================Process CSV============================#
def process_questions_from_csv(db, csv_path: str, output_csv: str = 'submission_file.csv'):
    df = pd.read_csv(csv_path)
    total_questions = len(df)
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "answer", "context", "ref_page"])
        writer.writeheader()
    
    for idx, row in enumerate(tqdm(df.iterrows(), total=total_questions, desc="Traitement des questions", unit="question")):
        qid = row[1]["id"]
        question = row[1]["question"]
        txt, meta = get_relevant_passage(question, db, n_results=3)
        if not txt:
            ans = "Info non trouvée"
            ctx = "Aucun contexte extrait"
            pg = "N/A"
        else:
            ans = get_gemini_response(question, txt)
            ctx = '["'+txt.replace('\n', ' ').replace('\r', ' ').strip('"')+'"]'
            pg = meta["page_number"]
        with open(output_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "question", "answer", "context", "ref_page"])
            writer.writerow({
                "id": qid,
                "question": question,
                "answer": ans,
                "context": ctx,
                "ref_page": str(pg)
            })
    with open(output_csv, 'rb+') as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size > 0:
            f.seek(size - 1)
            last_char = f.read(1)
            if last_char == b"\n":
                f.seek(size - 1)
                f.truncate()

#============================Main============================#
data, metadata = load_pdf(file_path="./data/MESUPRES_en_chiffres_MAJ.pdf")
tbls = extract_tables(file="./data/MESUPRES_en_chiffres_MAJ.pdf")
coll_name = 'rag'
db = load_data(documents=data, metadatas=metadata, collection_name=coll_name, tables=tbls)
process_questions_from_csv(db, './data/questions.csv')
#============================Main============================#