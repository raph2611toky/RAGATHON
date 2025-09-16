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

#============================Load PDF============================#
def load_pdf(file_path: str) -> Tuple[List[str], List[Dict]]:
    docs = []
    meta = []
    logical_page = 1
    with pdfplumber.open(file_path) as pdf:
        for phys_num, page in enumerate(tqdm(pdf.pages, desc=f"Lecture de {os.path.basename(file_path)}"), start=1):
            width = page.width
            height = page.height
            
            for half_num, bbox in enumerate([(0, 0, width / 2, height), (width / 2, 0, width, height)]):
                crop = page.crop(bbox)
                content = []
                
                tables = crop.find_tables()
                table_positions = sorted(tables, key=lambda t: t.bbox[1])
                
                prev_bottom = bbox[1] 
                for idx, table in enumerate(table_positions):
                    rel_top = prev_bottom - bbox[1]
                    rel_bottom = table.bbox[1] - bbox[1]
                    if rel_bottom > rel_top: 
                        above_bbox_rel = (0, rel_top, crop.width, rel_bottom)
                        above_crop = crop.crop(above_bbox_rel, relative=True)
                        above_text = above_crop.extract_text() or ""
                        if above_text.strip():
                            above_text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\{\}\'\"\«\»]', '', above_text)
                            above_text = re.sub(r'\s+', ' ', above_text).strip().replace('\n', ' ').replace('\r', ' ')
                            content.append(above_text)
                    else:
                        above_text = ""

                    table_data = table.extract()
                    table_str = table_to_text(table_data)
                    if table_str:
                        content.append(table_str)
                    
                    prev_bottom = table.bbox[3]
                
                rel_top = prev_bottom - bbox[1]
                rel_bottom = height - bbox[1] 
                if rel_bottom > rel_top:
                    below_bbox_rel = (0, rel_top, crop.width, rel_bottom)
                    below_crop = crop.crop(below_bbox_rel, relative=True)
                    below_text = below_crop.extract_text() or ""
                    if below_text.strip():
                        below_text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\{\}\'\"\«\»]', '', below_text)
                        below_text = re.sub(r'\s+', ' ', below_text).strip().replace('\n', ' ').replace('\r', ' ')
                        content.append(below_text)
                else:
                    below_text = ""
                
                if not table_positions:
                    full_text = crop.extract_text() or ""
                    if full_text.strip():
                        full_text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\{\}\'\"\«\»]', '', full_text)
                        full_text = re.sub(r'\s+', ' ', full_text).strip().replace('\n', ' ').replace('\r', ' ')
                        content.append(full_text)
                
                combined = "\n\n".join(content).strip()
                if combined:
                    docs.append(combined)
                    meta.append({
                        "filename": os.path.basename(file_path),
                        "physical_page": phys_num,
                        "half": "left" if half_num == 0 else "right",
                        "logical_page": logical_page
                    })
                    logical_page += 1
    
    return docs, meta

#============================Table to Text============================#
def table_to_text(tbl: List[List[str]]) -> str:
    if not tbl or all(all(not cell for cell in row) for row in tbl):
        return ""
    headers = tbl[0]
    md = "| " + " | ".join(str(cell or "") for cell in headers) + " |\n"
    md += "|---" * len(headers) + "|\n"
    for row in tbl[1:]:
        md += "| " + " | ".join(str(cell or "") for cell in row) + " |\n"
    return "Tableau (format markdown):\n" + md

#============================Load Data============================#
def load_data(documents: List[str], metadatas: List[Dict], collection_name: str):
    # Removed tables param since now integrated
    client = chromadb.EphemeralClient()
    embed = SentenceTransformerEmbeddingFunction(model_name="distiluse-base-multilingual-cased-v1")
    coll = client.create_collection(
        name=collection_name, embedding_function=embed
    )
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
    print("Getting relevant passage....")
    res = db.query(query_texts=[query], n_results=n_results, include=["documents", "metadatas"])
    print(res)
    candidates = res['documents'][0]
    metas = res['metadatas'][0]

    scored = [(similarity_score(query, doc), doc, meta) for doc, meta in zip(candidates, metas)]
    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best_doc, best_meta = scored[0]
    print(best_doc)
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

#============================Main============================#
data, metadata = load_pdf(file_path="./data/MESUPRES_en_chiffres_MAJ.pdf")
# print("#"*25)
# print(*data, sep=f'\n{"="*25}\n', end="\n")
# print("\n"+"#"*25+"\n")
coll_name = 'rag'
db = load_data(documents=data, metadatas=metadata, collection_name=coll_name)
print(get_relevant_passage("Quel était le nombre d'étudiants inscrits dans la région Vatovavy Fitovinany (privé) en 2022 ?", db, 2))
#============================Main============================#