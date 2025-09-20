#============================Imports============================#
from dotenv import load_dotenv, find_dotenv
from typing import List, Tuple, Dict, Optional
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
import warnings, unicodedata
import time
import json, random
import google.api_core.exceptions

#============================Config============================#
logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv(find_dotenv())
api_keys = [
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_API_KEY_SECOND")
]
api_keys = [key for key in api_keys if key]
if not api_keys:
    raise ValueError("No valid Gemini API keys found in environment variables")

os.makedirs("../submissions", exist_ok=True)


# ============================ Handle Merged Cells ============================ #
def fill_merged_cells(tbl: List[List[Optional[str]]]) -> List[List[Optional[str]]]:
    """
    Fill merged cells by propagating values horizontally and vertically.
    This handles common merged cell issues in PDF tables.
    """
    num_rows = len(tbl)
    if num_rows == 0:
        return []

    num_cols = max(len(row) for row in tbl)

    # Pad rows with None
    for row in tbl:
        row += [None] * (num_cols - len(row))

    # Fill horizontal merges (left to right)
    for r in range(num_rows):
        for c in range(1, num_cols):
            if tbl[r][c] is None and tbl[r][c - 1] is not None:
                tbl[r][c] = tbl[r][c - 1]

    # Fill vertical merges (top to bottom)
    for c in range(num_cols):
        for r in range(1, num_rows):
            if tbl[r][c] is None and tbl[r - 1][c] is not None:
                tbl[r][c] = tbl[r - 1][c]

    return tbl


# ============================ Table to Text ============================ #
def table_to_text(tbl: List[List[Optional[str]]]) -> str:
    """
    Convertit un tableau en markdown, avec largeur uniforme par colonne.
    Improved to remove empty rows and handle cleaned cells better.
    """
    if not tbl:
        return ""

    # Clean cells: replace None with "", flatten newlines, strip whitespace
    cleaned_tbl = [[str(cell or "").replace("\n", " ").strip() for cell in row] for row in tbl]

    # Remove entirely empty rows
    cleaned_tbl = [row for row in cleaned_tbl if any(cell for cell in row)]

    if not cleaned_tbl:
        return ""

    num_cols = max(len(row) for row in cleaned_tbl)
    col_widths = [0] * num_cols
    for row in cleaned_tbl:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Assume first row is headers
    headers = cleaned_tbl[0] + [""] * (num_cols - len(cleaned_tbl[0]))
    md = "| " + " | ".join(headers[i].ljust(col_widths[i]) for i in range(num_cols)) + " |\n"
    md += "| " + " | ".join("-" * col_widths[i] for i in range(num_cols)) + " |\n"

    for row in cleaned_tbl[1:]:
        padded_cells = [(row[i] if i < len(row) else "").ljust(col_widths[i]) for i in range(num_cols)]
        md += "| " + " | ".join(padded_cells) + " |\n"

    return md


# ============================ Texte (nouvelle version) ============================ #
def clean_text(text: str) -> str:
    """Nettoie et normalise du texte extrait (préserve accents et sauts de ligne)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)  # garder accents
    # garder les sauts de ligne mais supprimer espaces multiples
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


# ============================ Load PDF ============================ #
def load_pdf(file_path: str) -> Tuple[List[str], List[Dict]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    docs = []
    meta = []
    # Improved table settings for better detection accuracy
    # Changed 'keep_blank_chars' to 'text_keep_blank_chars' for compatibility with pdfplumber versions >= 0.8.0
    table_settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "intersection_tolerance": 3,
        "text_keep_blank_chars": False,
    }

    with pdfplumber.open(file_path) as pdf:
        for phys_num, page in enumerate(tqdm(pdf.pages, desc=f"Lecture de {os.path.basename(file_path)}"), start=1):
            width, height = page.width, page.height
            for half_num, bbox in enumerate([(0, 0, width / 2, height), (width / 2, 0, width, height)]):
                crop = page.crop(bbox)
                content = []
                tables = crop.find_tables(table_settings=table_settings)
                table_positions = sorted(tables, key=lambda t: t.bbox[1]) if tables else []
                prev_bottom = 0  # relatif au crop

                for table in table_positions:
                    rel_top = prev_bottom
                    rel_bottom = table.bbox[1]

                    # texte au-dessus du tableau
                    if rel_bottom > rel_top:
                        above_crop = crop.crop((0, rel_top, crop.width, rel_bottom), relative=True)
                        above_text = clean_text(above_crop.extract_text() or "")
                        if above_text:
                            content.append(above_text)

                    # tableau with merged cells handled
                    table_data = table.extract()
                    table_data = fill_merged_cells(table_data)
                    table_str = table_to_text(table_data)
                    if table_str:
                        content.append(table_str)

                    prev_bottom = table.bbox[3]

                # texte en dessous du dernier tableau
                rel_top = prev_bottom
                rel_bottom = crop.height
                if rel_bottom > rel_top:
                    below_crop = crop.crop((0, rel_top, crop.width, rel_bottom), relative=True)
                    below_text = clean_text(below_crop.extract_text() or "")
                    if below_text:
                        content.append(below_text)

                # cas sans tableau → texte brut
                if not table_positions:
                    full_text = clean_text(crop.extract_text() or "")
                    if full_text:
                        content.append(full_text)

                combined = "\n\n".join(content).strip()
                if combined:
                    # Replace "Tableau xx :" with "Titre du Tableau xx :"
                    combined = re.sub(r'Tableau (\d+)\s*:', r'Titre de la Tableau \1 :', combined)
                    # Replace "Graphe xx :" with "Titre du Graphe xx :"
                    combined = re.sub(r'Graphe (\d+)\s*:', r'Titre de la Graphe \1 :', combined)
                    docs.append(combined)
                    meta.append({
                        "filename": os.path.basename(file_path),
                        "physical_page": phys_num,
                        "half": "left" if half_num == 0 else "right"
                    })
    return docs, meta

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

def get_relevant_passage(query: str, db, n_results=1):
    # print("=======================================\n")
    # print("Getting relevant passage...")
    # print(query, end="\n__________________________\n")
    results = db.query(query_texts=[query], n_results=n_results, include=["documents", "metadatas"])
    top_ = [(d,m) for d, m in zip(results['documents'][0], results['metadatas'][0]) ]
    # print(top_, end="="*45 + "\n")
    return top_
# def get_relevant_passage(query: str, db, n_results=3):
#     res = db.query(query_texts=[query], n_results=5, include=["documents", "metadatas"])
#     candidates = res['documents'][0]
#     metas = res['metadatas'][0]

#     if not candidates:
#         return []

#     # Utiliser Gemini 1.5-flash pour re-ranker et sélectionner les meilleurs indices
#     # Préparer le prompt pour le re-ranking
#     combined_candidates = "\n\n".join([f"# Doc {i}: {candidates[i]}" for i in range(len(candidates))])
#     rerank_prompt = f"""
#     Vous êtes un expert en sélection de passages pertinents. Analysez les candidats suivants et sélectionnez les {n_results} indices des documents les plus pertinents pour la question : '{query}'.
#     Choisissez les indices (nombres entre 0 et {len(candidates)-1}) des documents qui contiennent la réponse exacte ou la plus proche.
#     Si moins de {n_results} sont pertinents, retournez seulement ceux qui le sont.
#     Format de réponse : {{"best_indices": [<nombre1>, <nombre2>, ...]}}
#     Candidats :
#     {combined_candidates}
#     """

#     model = genai.GenerativeModel("gemini-1.5-flash")
#     max_retries = 3
#     retry_delay = 5
#     best_indices = [0]  # Default
#     for key_index, api_key in enumerate(api_keys):
#         try:
#             genai.configure(api_key=api_key)
#             for attempt in range(max_retries):
#                 try:
#                     res = model.generate_content(rerank_prompt)
#                     time.sleep(0.1)
#                     response_text = res.text.strip() if res.text else ""
#                     if not response_text:
#                         raise ValueError("Empty response text")
#                     json_match = regex.search(r'\{(?:[^{}]|(?R))*\}', response_text, regex.DOTALL)
#                     if json_match:
#                         json_str = json_match.group(0).replace(",}", "}").replace(",]", "]").replace("\n", "  ")
#                         rerank_data = json.loads(json_str)
#                         best_indices = rerank_data.get("best_indices", [0])
#                         best_indices = [int(idx) for idx in best_indices if idx < len(candidates)]
#                     break
#                 except (google.api_core.exceptions.ResourceExhausted, json.JSONDecodeError, ValueError) as e:
#                     if attempt < max_retries - 1:
#                         time.sleep(retry_delay)
#                         retry_delay *= 1.1
#                     else:
#                         break
#             break
#         except Exception as e:
#             continue

#     # Retourner les top n_results (ou moins si pas assez)
#     top_ = [(candidates[idx], metas[idx]) for idx in best_indices[:n_results]]
#     return top_

#============================Classify Question============================#
def classify_q(q: str) -> str:
    q = q.lower()
    if "nombre" in q or "effectif" in q:
        return "num"
    elif "pourcentage" in q or "%" in q or "proportion" in q:
        return "pct"
    elif "taux de " in q:
        return "success"
    elif "total" in q or "combien" in q:
        return "num"
    return "gen"

#============================Extract Number============================#
def extract_num(ans: str, q_type: str) -> str:
    # Assurer que ans est une string
    ans = str(ans) if ans is not None else ""
    ans_ = ans  # Copie originale

    ans = re.sub(r'\s+', '', ans)
    
    if not ans:
        return "Information non trouvée"
    
    if q_type == "num" and ans.isnumeric():
        return ans
    
    cleaned_ans = ans.strip()
    if (q_type in ["pct", "success"]) and cleaned_ans.isnumeric():
        value = int(cleaned_ans)
        if value > 100:
            numbers = re.findall(r'\b\d+\b(?=%|\s)', ans)
            valid_percentages = [num for num in numbers if int(num) <= 100]
            return f"{valid_percentages[0]}%" if valid_percentages else "Information non trouvée"
        return f"{cleaned_ans}%"
    
    numbers = re.findall(r'\b\d+\b', re.sub(r'\s+', '', ans))
    if not numbers:
        return "Information non trouvée"
    
    years = [num for num in numbers if len(num) == 4 and 1900 <= int(num) <= 2100]
    non_years = [num for num in numbers if num not in years and num!="0"]
    
    if non_years:
        if q_type == "num":
            return non_years[0] if len(non_years) == 1 else non_years[-1]
        elif q_type in ["pct", "success"]:
            valid_values = [num for num in non_years if int(num) <= 100]
            return f"{valid_values[0]}%" if valid_values else non_years[0]
    return numbers[0] if numbers else "Information non trouvée"

#============================RAG Prompt============================#
def make_rag_prompt(query: str, contexts: List[Tuple[str, Dict]]) -> str:
    combined_context = "\n\n".join([f"# ============== Doc {doci} ============== #\n{contexts[doci][0]}\n# ============== Doc {doci} ============== #" for doci in range(len(contexts))]) 
    q_type = classify_q(query)
    instructions = ""
    if q_type == "num":
        instructions = "La reponse doit etre un nombre,faite des calculs si ca n'existe pas mais ca doit etre un nombre (sans texte supplémentaire: ex: 1254)."
    elif q_type == "pct":
        instructions = "La reponse est un pourcentage de nombre, caculs si ca n'existe pas (sans texte supplémentaire, ex: 45%)."
    elif q_type == "success":
        instructions = "La réponse est un taux de réussite, retournez uniquement le taux exact (sans texte supplémentaire, ex: 85%)."
    format = """{"answer":"<reponse>","doc_index":<nombre_entier>}"""
    return f"""
    Expert en stats éducatives Madagascar. Analysez les contextes fournis pour répondre à la question. Retournez une réponse contenant 'answer' (réponse directe) et 'doc_index' (nombre d'index de la document dans le contexte). Soyez précis, concis, factuel. Pas d'info hors contexte. {instructions}
    Choisissez le document le plus pertinent, pas une liste. Si pas de réponse précise, utilisez l'approximation si disponible (ex: 'jusqu'à 85%' ou 'entre 2016 à 2020' pour 2018), faites les calculs si besoin meme et donner directe le resultat finale sans les calculs ni explication.
    Et forcement, il y a une reponse dans le contexte fournie, alors ne laisser aucune vide de ces json. Si vraiment rien, utilisez "Information non trouvée" pour answer.
    **Format attendu**: {format}
    **Question**: {query}
    **Contexts**: {combined_context}
    **Réponse**:
    """

#============================Gemini Response============================#
def get_gemini_response(query: str, contexts: List[Tuple[str, Dict]]) -> Dict:
    prompt = make_rag_prompt(query, contexts)
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
                    response_text = res.text.strip() if res.text else ""
                    if not response_text:
                        raise ValueError("Empty response text")
                    json_match = regex.search(r'\{(?:[^{}]|(?R))*\}', response_text, regex.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        json_str = json_str.replace(",}", "}").replace(",]", "]")
                        json_str = json_str.replace("\n", "  ")
                        try:
                            response_data = json.loads(json_str)
                            # Convertir en strings pour éviter les erreurs d'attribut
                            if "answer" in response_data:
                                response_data["answer"] = str(response_data["answer"]) if response_data["answer"] is not None else "Information non trouvée"
                            if "doc_index" in response_data:
                                response_data["doc_index"] = str(response_data["doc_index"]) if response_data["doc_index"] is not None else "0"
                        except json.JSONDecodeError as e:
                            answer_match = re.search(r'answer:?\s*([^\n]+)', response_text, re.IGNORECASE)
                            context_match = re.search(r'"doc_index":?\s*([^\n]+)', response_text, re.IGNORECASE)
                            response_data = {
                                "answer": str(answer_match.group(1).strip()) if answer_match else "Information non trouvée",
                                "doc_index": str(context_match.group(1).strip()) if context_match else "0"
                            }
                    else:
                        answer_match = re.search(r'answer:?\s*([^\n]+)', response_text, re.IGNORECASE)
                        context_match = re.search(r'doc_index:?\s*([^\n]+)', response_text, re.IGNORECASE)
                        response_data = {
                            "answer": str(answer_match.group(1).strip()) if answer_match else "Information non trouvée",
                            "doc_index": str(context_match.group(1).strip()) if context_match else "0"
                        }
                    q_type = classify_q(query)
                    if q_type in ["num", "pct", "success"] and "answer" in response_data:
                        extracted_value = extract_num(response_data["answer"], q_type)
                        if extracted_value:
                            response_data["answer"] = extracted_value
                    # Assurer que answer n'est pas None ou 0
                    if response_data["answer"] in [None, "0", "None"]:
                        response_data["answer"] = "Information non trouvée"
                    return response_data
                except (google.api_core.exceptions.ResourceExhausted, json.JSONDecodeError, ValueError) as e:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 1.1
                    else:
                        break
        except Exception as e:
            print(f"API configuration error: {str(e)}")
            continue
    return {"answer": "Information non trouvée", "doc_index": "0"}

#============================Process CSV============================#
def process_questions_from_csv(db, csv_path: str, output_csv: str = '../submissions/submission_file_base.csv'):
    try:
        df = pd.read_csv(csv_path)
        total_questions = len(df)
        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "question", "answer", "context", "ref_page"])
            writer.writeheader()
        for idx, row in enumerate(tqdm(df.iterrows(), total=total_questions, desc="Traitement des questions", unit="question")):
            qid = row[1]["id"]
            question = row[1]["question"]
            passages = get_relevant_passage(question, db, n_results=3)
            if not passages:
                print("❗ Pas de passages...", passages)
                ans = "Information non trouvée"
                ctx = "Aucun contexte extrait"
                pg = "N/A"
            else:
                response = get_gemini_response(question, passages)
                ans = response.get("answer", "Information non trouvée")
                doc_index = response.get("doc_index", "0")
                doc_index = doc_index if str(doc_index).isnumeric() and int(doc_index) < len(passages) else "0"
                ctx = re.sub(r'[\r\n]+', '  ', passages[int(doc_index)][0])
                meta = passages[int(doc_index)][1]
                pg = meta.get("physical_page", 27)
                q_type = classify_q(question)
                if q_type in ["num", "pct", "success"]:
                    extracted_value = extract_num(ans, q_type)
                    if extracted_value:
                        ans = extracted_value
            with open(output_csv, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "question", "answer", "context", "ref_page"])
                writer.writerow({
                    "id": qid,
                    "question": question,
                    "answer": ans if ans not in ["None", "0"] else "Information non trouvée",
                    "context": '["' + ctx.strip('"') + '"]' if ctx else '["Aucun contexte"]',
                    "ref_page": pg if pg else "N/A"
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
    data, metadata = load_pdf(file_path="../data/MESUPRES_en_chiffres_MAJ.pdf")
    coll_name = 'rag'
    db = load_data(documents=data, metadatas=metadata, collection_name=coll_name)
    process_questions_from_csv(db, '../data/questions.csv')
except Exception as e:
    print(f"Main execution error: {str(e)}")
#============================Main============================#