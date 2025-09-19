from dotenv import load_dotenv, find_dotenv
from typing import List, Tuple, Dict
import pdfplumber
from tqdm import tqdm
import google.generativeai as genai
import chromadb
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
import pandas as pd
import csv
import re, logging, regex
import difflib, traceback
import os
import warnings
import time
import json, random
import google.api_core.exceptions

#============================Config============================#
logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv(find_dotenv())
api_keys = [
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_API_KEY_SECOND"),
]
api_keys = [key for key in api_keys if key]
if not api_keys:
    raise ValueError("No valid Gemini API keys found in environment variables")

os.makedirs("../submissions", exist_ok=True)

#============================Load PDF============================#
def load_pdf(file_path: str) -> Tuple[List[str], List[Dict]]:
    docs = []
    meta = []
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
                            above_text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\{\}\'\"\Â«\Â»]', '', above_text)
                            above_text = re.sub(r'\s+', ' ', above_text).strip().replace('\n', ' ').replace('\r', ' ')
                            content.append(above_text)
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
                        below_text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\{\}\'\"\Â«\Â»]', '', below_text)
                        below_text = re.sub(r'\s+', ' ', below_text).strip().replace('\n', ' ').replace('\r', ' ')
                        content.append(below_text)
                if not table_positions:
                    full_text = crop.extract_text() or ""
                    if full_text.strip():
                        full_text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\{\}\'\"\Â«\Â»]', '', full_text)
                        full_text = re.sub(r'\s+', ' ', full_text).strip().replace('\n', ' ').replace('\r', ' ')
                        content.append(full_text)
                combined = "\n\n".join(content).strip()
                if combined:
                    docs.append(combined)
                    meta.append({
                        "filename": os.path.basename(file_path),
                        "physical_page": phys_num,
                        "half": "left" if half_num == 0 else "right"
                    })
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
def load_data(documents, metadatas, collection_name):
    client = chromadb.EphemeralClient()
    google_api_key = os.getenv("GEMINI_API_KEY_SECOND")
    if not google_api_key:
        raise ValueError("GEMINI_API_KEY_SECOND not found in environment variables")
    embedding_function = GoogleGenerativeAiEmbeddingFunction(api_key=google_api_key)
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)
    count = collection.count()
    print(f"Collection already contains {count} documents")
    ids = [str(i) for i in range(count, count + len(documents))]
    for i in tqdm(range(0, len(documents), 100), desc="Adding documents", unit_scale=100):
        collection.add(
            ids=ids[i:i + 100],
            documents=documents[i:i + 100],
            metadatas=metadatas[i:i + 100],
        )
    print(f"Documents loaded successfully")
    return collection

#============================Get DB============================#
def get_db(collection_name):
    google_api_key = os.getenv("GEMINI_API_KEY_SECOND")
    if not google_api_key:
        raise ValueError("GEMINI_API_KEY_SECOND not found in environment variables")
    client = chromadb.EphemeralClient()
    embedding_function = GoogleGenerativeAiEmbeddingFunction(api_key=google_api_key, task_type="RETRIEVAL_QUERY")
    db = client.get_collection(name=collection_name, embedding_function=embedding_function)
    return db

#============================Get Relevant Passage============================#
def get_relevant_passage(query: str, db, n_results=1):
    results = db.query(query_texts=[query], n_results=n_results, include=["documents", "metadatas"])
    return results['documents'][0][0], results['metadatas'][0][0]

#============================Classify Question============================#
def classify_q(q: str) -> str:
    q = q.lower()
    if "nombre" in q or "effectif" in q or "combien" in q:
        return "num"
    elif "pourcentage" in q or "%" in q or "proportion"in q:
        return "pct"
    elif "taux" in q:
        return "success"
    elif "total" in q or "ratio":
        return "num"
    return "gen"

#============================Extract Number============================#
def extract_num(ans: str, q_type: str) -> str:
    ans, ans_ = [ans.strip()]*2
    ans = re.sub(r'\s+', '', ans)
    if q_type == "num" and ans.isnumeric():
        return f"{ans:,}".replace(","," ")
    if (q_type in ["pct", "success"]) and ans.isnumeric():
        value = int(ans)
        if value > 100:
            numbers = re.findall(r'\b\d+\b(?=%|\s)', ans)
            valid_percentages = [num for num in numbers if int(num) <= 100]
            return f"{valid_percentages[0]}%" if valid_percentages else ans_
        return f"{ans.replace(",",".")}%"
    numbers = re.findall(r'\b\d+\b', re.sub(r'\s+', '', ans))
    if not numbers:
        return ans
    years = [num for num in numbers if len(num) == 4 and 1900 <= int(num) <= 2100]
    non_years = [num for num in numbers if num not in years]
    if non_years:
        if q_type == "num":
            return f"{non_years[0]:,}".replace(','," ") if len(non_years) == 1 else f"{non_years[-1]:,}".replace(','," ")
        elif q_type in ["pct", "success"]:
            valid_values = [num for num in non_years if int(num) <= 100]
            return f"{valid_values[0]}%" if valid_values else non_years[0]
    return str(numbers[0]) if numbers else ans

#============================RAG Prompt============================#
def make_rag_prompt(query: str, relevant_passage: str) -> str:
    cleaned = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    q_type = classify_q(query)
    instructions = ""
    if q_type == "num":
        instructions = "Répondez uniquement avec un nombre (ex: 1254)."
    elif q_type == "pct":
        instructions = "Répondez uniquement avec un pourcentage (ex: 45%)."
    elif q_type == "success":
        instructions = "Répondez uniquement avec un taux de quelque chose (ex: 85%)."
    prompt = f"""Expert en stats éducatives Madagascar. Analysez le contexte fourni pour répondre à la question. Réponse directe, précise, factuelle, sans texte superflu. {instructions}
    Si pas de réponse exacte, utilisez une approximation si disponible (ex: 'jusqu'à 85%' ou 'entre 2016 à 2020' pour 2018), faites les calculs si nécessaire et donnez directement le résultat final.
    QUESTION: {query}
    PASSAGE: {cleaned}
    ANSWER:"""
    return prompt

#============================Gemini Response============================#
def get_gemini_response(query: str, passage: str) -> Dict:
    prompt = make_rag_prompt(query, passage)
    model = genai.GenerativeModel("gemini-2.5-flash")
    max_retries = 3
    retry_delay = 5
    for key_index, api_key in enumerate(api_keys):
        try:
            genai.configure(api_key=api_key)
            for attempt in range(max_retries):
                try:
                    res = model.generate_content(prompt)
                    time.sleep(0.1)
                    response_text = res.text.strip()
                    q_type = classify_q(query)
                    extracted_value = extract_num(response_text, q_type) if q_type in ["num", "pct", "success"] else response_text
                    return {"answer": extracted_value}
                except (google.api_core.exceptions.ResourceExhausted, json.JSONDecodeError) as e:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 1.1
                    else:
                        break
        except Exception as e:
            print(f"API configuration error with key {key_index + 1}: {str(e)}")
            continue
    return {"answer": "Erreur : Toutes les clés API ont dépassé leur quota ou ont échoué"}

#============================Process CSV============================#
def process_questions_from_csv(db, csv_path: str, output_csv: str = '../submissions/submission_rag.csv'):
    try:
        df = pd.read_csv(csv_path)
        total_questions = len(df)
        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "question", "answer", "context", "ref_page"])
            writer.writeheader()
        for idx, row in enumerate(tqdm(df.iterrows(), total=total_questions, desc="Traitement des questions", unit="question")):
            qid = row[1]["id"]
            question = row[1]["question"]
            passage, meta = get_relevant_passage(question, db, n_results=1)
            if not passage:
                ans = "Info non trouvée"
                ctx = "Aucun contexte extrait"
                pg = "N/A"
            else:
                response = get_gemini_response(question, passage)
                ans = response.get("answer", "Erreur de traitement")
                ctx = passage
                pg = meta.get("physical_page", 27)
            with open(output_csv, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "question", "answer", "context", "ref_page"])
                ans = re.sub(r'[\r\n]+', '  ', ans)
                ctx = re.sub(r'[\r\n]+', '  ', ctx)
                writer.writerow({
                    "id": qid,
                    "question": question,
                    "answer": ans.replace(',', ' '),
                    "context": '["' + ctx.replace(',',' ') + '"]',
                    "ref_page": pg
                })
    except Exception as e:
        print(f"Error in process_questions_from_csv: {str(e)}")
        print(traceback.format_exc())
    finally:
        with open(output_csv, 'rb+') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size > 0:
                f.seek(size - 1)
                last_char = f.read(1)
                if last_char in [b"\n", b"\r"]:
                    f.seek(size - 1)
                    f.truncate()

#============================Main============================#
try:
    data, metadata = load_pdf(file_path="../data/MESUPRES_en_chiffres_MAJ.pdf")
    coll_name = 'rag'
    db = load_data(documents=data, metadatas=metadata, collection_name=coll_name)
    db = get_db(coll_name)
    process_questions_from_csv(db, '../data/questions.csv')
except Exception as e:
    print(f"Main execution error: {str(e)}")