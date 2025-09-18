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
    client = chromadb.EphemeralClient()
    embed = SentenceTransformerEmbeddingFunction(model_name="distiluse-base-multilingual-cased-v1")
    coll = client.create_collection(name=collection_name, embedding_function=embed)
    coll.add(documents=documents, metadatas=metadatas, ids=[f"doc_{i}" for i in range(len(documents))])
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

def get_relevant_passage(query: str, db, n_results=3):
    res = db.query(query_texts=[query], n_results=n_results, include=["documents", "metadatas"])
    candidates = res['documents'][0]
    metas = res['metadatas'][0]
    scored = [(similarity_score(query, doc), doc, meta) for doc, meta in zip(candidates, metas)]
    scored.sort(key=lambda x: x[0], reverse=True)
    top_ = [(doc, meta) for _, doc, meta in scored[:n_results]]
    return top_

#============================Classify Question============================#
def classify_q(q: str) -> str:
    q = q.lower()
    if "nombre" in q or "effectif" in q or "combien" in q:
        return "num"
    elif "pourcentage" in q or "%" in q:
        return "pct"
    elif "taux de réussite" in q:
        return "success"
    return "gen"

#============================Extract Number============================#
def extract_num(ans: str, q_type: str) -> str:
    # print("Extraction de nombre....", q_type, ans)
    ans, ans_ = [ans.strip()]*2
    ans = re.sub(r'\s+', '', ans)
    
    if q_type == "num" and ans.isnumeric():
        return ans
    
    cleaned_ans = ans.strip()
    if (q_type in ["pct", "success"]) and cleaned_ans.isnumeric():
        value = int(cleaned_ans)
        if value > 100:
            numbers = re.findall(r'\b\d+\b(?=%|\s)', ans)
            valid_percentages = [num for num in numbers if int(num) <= 100]
            return f"{valid_percentages[0]}%" if valid_percentages else ans_
        return f"{cleaned_ans}%"
    
    numbers = re.findall(r'\b\d+\b', re.sub(r'\s+', '', ans))
    if not numbers:
        return ans
    
    years = [num for num in numbers if len(num) == 4 and 1900 <= int(num) <= 2100]
    non_years = [num for num in numbers if num not in years]
    
    if non_years:
        if q_type == "num":
            return non_years[0] if len(non_years) == 1 else non_years[-1]
        elif q_type in ["pct", "success"]:
            valid_values = [num for num in non_years if int(num) <= 100]
            return f"{valid_values[0]}%" if valid_values else non_years[0]
    return numbers[0] if numbers else ans

#============================RAG Prompt============================#
def make_rag_prompt(query: str, contexts: List[Tuple[str, Dict]]) -> str:
    combined_context = "\n\n".join([f"# ============== Doc {doci} ============== #\n{contexts[doci][0]}\n# ============== Doc {doci} ============== #" for doci in range(len(contexts))]) 
    q_type = classify_q(query)
    instructions = ""
    if q_type == "num":
        instructions = "La reponse doit etre un nombre, trouve des similarité ou faite des calculs si ca n'existe pas mais ca doit etre un nombre (sans texte supplémentaire: ex: 1254)."
    elif q_type == "pct":
        instructions = "La reponse est un pourcentage de nombre, caculs si ca n'existe pas en priorisant la vraie reponse mais donne toujours un pourcentage valide (sans texte supplémentaire, ex: 45%)."
    elif q_type == "success":
        instructions = "La réponse est un taux de réussite, retournez uniquement le taux exact en priorisant la vraie reponse et si jamais, tu ne trouve pas de reonse exacte, trouve des similarité ou faire des calculs (sans texte supplémentaire, ex: 85%)."
    format = """{"answer":"<reponse>","doc_index":<nombre_entier_qui_indique_l_index_de_doc_entre_les_contextes_ou_le_plus_pertinent_jamais_null_par_defaut_c_est_0>}"""
    return f"""
    Expert en stats éducatives Madagascar. Analysez les contextes fournis pour répondre à la question. Retournez une réponse contenant 'answer' (réponse directe) et 'relevant_context' (le contexte exact qui répond à la question, y compris sa métadonnée). Soyez précis, concis, factuel. Pas d'info hors contexte. {instructions}
    Choisissez le document le plus pertinent, pas une liste. Si pas de réponse précise, utilisez l'approximation si disponible (ex: 'jusqu'à 85%' ou 'entre 2016 à 2020' pour 2018), faites les calculs si besoin meme et donner directe le resultat finale sans les calculs ni explication.
    Et forcement, il y a une reponse dans le contexte fournie, alors ne laisser aucune vide de ces json.
    **Format attendu**: {format}
    **Question**: {query}
    **Contexts**: {combined_context}
    **Réponse**:
    """

#============================Gemini Response============================#
def get_gemini_response(query: str, contexts: List[Tuple[str, Dict]]) -> Dict:
    # print("Prepare to get gemini response...")
    # print(contexts)
    prompt = make_rag_prompt(query, contexts)
    model = genai.GenerativeModel("gemini-2.5-flash-pro")
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
                    # print("Gemini response....")
                    # print(response_text)
                    json_match = regex.search(r'\{(?:[^{}]|(?R))*\}', response_text, regex.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        json_str = json_str.replace(",}", "}").replace(",]", "]")
                        json_str = json_str.replace("\n", "  ")
                        try:
                            response_data = json.loads(json_str)
                            # print("json response loaded...")
                        except json.JSONDecodeError as e:
                            # print(f"JSON parsing failed: {str(e)} - Raw JSON: {json_str}")
                            answer_match = re.search(r'answer:?\s*([^\n]+)', response_text, re.IGNORECASE)
                            context_match = re.search(r'"doc_index":?\s*([^\n]+)', response_text, re.IGNORECASE)
                            response_data = {
                                "answer": answer_match.group(1).strip() if answer_match else "Aucune réponse claire",
                                "doc_index": context_match.group(1).strip() if context_match else 0
                            }
                    else:
                        # print("No JSON structure detected in response")
                        answer_match = re.search(r'answer:?\s*([^\n]+)', response_text, re.IGNORECASE)
                        context_match = re.search(r'doc_index:?\s*([^\n]+)', response_text, re.IGNORECASE)
                        response_data = {
                            "answer": answer_match.group(1).strip() if answer_match else "Aucune réponse claire",
                            "doc_index": context_match.group(1).strip() if context_match else 0
                        }
                    q_type = classify_q(query)
                    if q_type in ["num", "pct", "success"] and "answer" in response_data:
                        extracted_value = extract_num(response_data["answer"], q_type)
                        if extracted_value:
                            response_data["answer"] = extracted_value
                    # print("Gemini response success...✔")
                    return response_data
                except (google.api_core.exceptions.ResourceExhausted, json.JSONDecodeError) as e:
                    # print(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 1.1
                    else:
                        break
        except Exception as e:
            # print(f"API configuration error: {str(e)}")
            continue
    return {"answer": "Erreur : Toutes les clés API ont dépassé leur quota ou ont échoué", "relevant_context": None}

#============================Process CSV============================#
def process_questions_from_csv(db, csv_path: str, output_csv: str = 'submission_file.csv'):
    try:
        df = pd.read_csv(csv_path)
        total_questions = len(df)
        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "question", "answer", "context", "ref_page"])
            writer.writeheader()
        for idx, row in enumerate(tqdm(df.iterrows(), total=total_questions, desc="Traitement des questions", unit="question")):
            qid = row[1]["id"]
            question = row[1]["question"]
            # print(f"\n#{'='*25}#\n")
            # print(f"Processing question {qid}: {question}")
            passages = get_relevant_passage(question, db, n_results=5)
            if not passages:
                ans = "Info non trouvée"
                ctx = "Aucun contexte extrait"
                pg = "N/A"
            else:
                response = get_gemini_response(question, passages)
                print(f"Response for {qid}: {response}")
                ans = response.get("answer", "Erreur de traitement")
                doc_index = response.get("doc_index", "0")
                doc_index = doc_index if str(doc_index).isnumeric() and int(doc_index)<len(passages) else 0
                ctx = re.sub(r'[\r\n]+', '  ', passages[int(doc_index)][0])
                meta = passages[int(doc_index)][1]
                pg = meta.get("physical_page", 27)
                q_type = classify_q(question)
                if q_type in ["num", "pct", "success"]:
                    extracted_value = extract_num(ans, q_type)
                    if extracted_value:
                        ans = extracted_value
                        # if q_type == "pct" and not ans.endswith("%"):
                        #     ans = f"{ans}%"
            # print("ready to write answer...")
            # print(f"\n#{'='*25}#\n")
            with open(output_csv, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "question", "answer", "context", "ref_page"])
                writer.writerow({
                    "id": qid,
                    "question": question,
                    "answer": ans,
                    "context": '["' + ctx.strip('"') + '"]',
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
                if last_char in [b"\n",b"\r"]:
                    f.seek(size - 1)
                    f.truncate()

#============================Main============================#
try:
    data, metadata = load_pdf(file_path="./data/MESUPRES_en_chiffres_MAJ.pdf")
    coll_name = 'rag'
    db = load_data(documents=data, metadatas=metadata, collection_name=coll_name)
    process_questions_from_csv(db, './data/questions.csv')
except Exception as e:
    print(f"Main execution error: {str(e)}")
#============================Main============================#