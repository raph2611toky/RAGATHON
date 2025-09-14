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
from PIL import Image
import io

#============================Config============================#
warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv(find_dotenv())
api_keys = [
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_API anyagrok_KEY_SECOND"),
    os.getenv("GEMINI_API_KEY_THIRD"),
    os.getenv("GEMINI_API_KEY_FOURTH"),
]
api_keys = [key for key in api_keys if key]
if not api_keys:
    raise ValueError("No valid Gemini API keys found in environment variables")

#============================Load PDF============================#
def load_pdf(file_path: str) -> Tuple[List[str], List[Dict]]:
    reader = PdfReader(file_path)
    docs = []
    meta = []
    for num, page in enumerate(tqdm(reader.pages, desc=f"Lecture de {os.path.basename(file_path)}"), start=1):
        text = page.extract_text()
        if text and text.strip():
            docs.append(text.strip().replace('\n', ' ').replace('\r', ' '))
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

#============================Extract Images============================#
def extract_images(file: str) -> List[Dict]:
    images = []
    with pdfplumber.open(file) as pdf:
        for pg, page in enumerate(tqdm(pdf.pages, desc="Extraction des images"), start=1):
            imgs = page.images
            for img in imgs:
                images.append({
                    "image": img,
                    "page_num": pg
                })
    return images

#============================Describe Image============================#
def describe_image(image_dict: Dict, pdf_file: str) -> str:
    with pdfplumber.open(pdf_file) as pdf:
        page = pdf.pages[image_dict["page_num"] - 1]
        genai.configure(api_key=api_keys[0])
        model = genai.GenerativeModel("gemini-1.5-flash")
        img_data = image_dict["image"]
        x1, y1, x2, y2 = img_data["x0"], img_data["top"], img_data["x1"], img_data["bottom"]
        img = page.crop((x1, y1, x2, y2)).to_image(resolution=300).original  # Get PIL.Image.Image
        res = model.generate_content(["Décrivez ce graphe ou cette image en détail, en vous concentrant sur les données, les étiquettes et les tendances :", img])
        return res.text.replace('\n', ' ').replace('\r', ' ')

#============================Table to Text============================#
def table_to_text(tbl: List[List[str]]) -> str:
    return " ".join([" ".join(str(cell or "") for cell in row) for row in tbl if row])

#============================Load Data============================#
def load_data(documents: List[str], metadatas: List[Dict], collection_name: str, tables: List[Dict], images: List[Dict], pdf_file: str):
    client = chromadb.EphemeralClient()
    embed = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
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
    for img in tqdm(images, desc="Ajout des descriptions d'images à la collection"):
        img_desc = describe_image(img, pdf_file)
        documents.append(img_desc)
        metadatas.append({
            "filename": "graph",
            "page_number": img["page_num"]
        })
    coll.add(
        documents=documents,
        metadatas=metadatas,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
    return coll

#============================Get Passage============================#
def get_relevant_passage(query: str, db, n_results=1):
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
        txt, meta = get_relevant_passage(question, db, n_results=1)
        if not txt:
            ans = "Info non trouvée"
            ctx = "Aucun contexte extrait"
            pg = "N/A"
        else:
            ans = get_gemini_response(question, txt)
            ctx = txt.replace('\n', ' ').replace('\r', ' ')
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
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b'\n':
            f.seek(-1, os.SEEK_END)
            f.truncate()

#============================Main============================#
data, metadata = load_pdf(file_path="./data/MESUPRES_en_chiffres_MAJ.pdf")
tbls = extract_tables(file="./data/MESUPRES_en_chiffres_MAJ.pdf")
images = extract_images(file="./data/MESUPRES_en_chiffres_MAJ.pdf")
coll_name = 'rag'
db = load_data(documents=data, metadatas=metadata, collection_name=coll_name, tables=tbls, images=images, pdf_file="./data/MESUPRES_en_chiffres_MAJ.pdf")
process_questions_from_csv(db, './data/questions.csv')
#============================Main============================#