import os
from dotenv import load_dotenv, find_dotenv
from typing import List, Tuple, Dict
import pdfplumber
from tqdm import tqdm
import re
from pypdf import PdfReader
import logging
import warnings, json

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


#============================Table to Text============================#
def table_to_text(tbl: List[List[str]]) -> str:
    """
    Converts a table to markdown text format.

    Args:
        tbl (List[List[str]]): Extracted table data.

    Returns:
        str: Markdown representation of the table.
    """
    if not tbl or all(all(not cell for cell in row) for row in tbl):
        return ""
    headers = tbl[0]
    md = "| " + " | ".join(str(cell or "") for cell in headers) + " |\n"
    md += "|---" * len(headers) + "|\n"
    for row in tbl[1:]:
        md += "| " + " | ".join(str(cell or "") for cell in row) + " |\n"
    return "Tableau (format markdown):\n" + md


#============================Load PDF============================#
def load_pdf(file_path: str) -> Tuple[List[str], List[Dict]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

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
                table_positions = sorted(tables, key=lambda t: t.bbox[1]) if tables else []
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

# =============================== LOAD DATA =============================== #


def load_pdf_init(file_path: str) -> Tuple[List[str], List[Dict]]:
    """
    Reads text content from a PDF file, returns page texts and metadata.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        Tuple[List[str], List[Dict]]: List of page texts and metadata with filename + page number.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    reader = PdfReader(file_path)
    documents = []
    metadatas = []

    for page_number, page in enumerate(
        tqdm(reader.pages, desc=f"Reading {os.path.basename(file_path)}"), start=1
    ):
        text = page.extract_text()
        if text and text.strip():
            documents.append(text.strip())
            metadatas.append({
                "filename": os.path.basename(file_path),
                "page_number": page_number
            })

    return documents, metadatas

def main():
    file_path = "../data/MESUPRES_en_chiffres_MAJ.pdf"
    documents, metadatas = load_pdf(file_path)
    with open("../data/load_pdf.md", "w") as f:
        f.write(f"# ======================================= DOCUMENTS ======================================= #\n{'\n---\n'.join(documents)}\n# ======================================= META ======================================= #{'\n---\n'.join([json.dumps(meta)for meta in metadatas])}")
    # documents, metadatas = load_pdf_init(file_path)
    # with open("../data/load_pdf_init.md", "w") as f:
    #     f.write(f"# ======================================= DOCUMENTS ======================================= #\n{'\n---\n'.join(documents)}\n# ======================================= META ======================================= #{'\n---\n'.join([json.dumps(meta)for meta in metadatas])}")

if __name__ == "__main__":
    main()