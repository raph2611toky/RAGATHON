import os
from dotenv import load_dotenv, find_dotenv
from typing import List, Tuple, Dict
import pdfplumber
from tqdm import tqdm
import re
from pypdf import PdfReader
import logging
import warnings, json
import unicodedata

# ============================ Config ============================ #
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


# ============================ Table to Text (ancien code) ============================ #
def table_to_text(tbl: List[List[str]]) -> str:
    """
    Convertit un tableau en markdown, avec largeur uniforme par colonne.
    """
    if not tbl or all(all(not cell for cell in row) for row in tbl):
        return ""

    cleaned_tbl = [[str(cell or "").replace("\n", " ").strip() for cell in row] for row in tbl]

    num_cols = max(len(row) for row in cleaned_tbl)
    col_widths = [0] * num_cols
    for row in cleaned_tbl:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    headers = cleaned_tbl[0]
    md = "| " + " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers))) + " |\n"
    md += "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |\n"

    for row in cleaned_tbl[1:]:
        padded_cells = [
            (row[i] if i < len(row) else "").ljust(col_widths[i]) for i in range(num_cols)
        ]
        md += "| " + " | ".join(padded_cells) + " |\n"

    return "Tableau (format markdown):\n" + md



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
    with pdfplumber.open(file_path) as pdf:
        for phys_num, page in enumerate(tqdm(pdf.pages, desc=f"Lecture de {os.path.basename(file_path)}"), start=1):
            width, height = page.width, page.height
            for half_num, bbox in enumerate([(0, 0, width / 2, height), (width / 2, 0, width, height)]):
                crop = page.crop(bbox)
                content = []
                tables = crop.find_tables()
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

                    # tableau
                    table_data = table.extract()
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
                    docs.append(combined)
                    meta.append({
                        "filename": os.path.basename(file_path),
                        "physical_page": phys_num,
                        "half": "left" if half_num == 0 else "right"
                    })
    return docs, meta

def main():
    file_path = "../data/MESUPRES_en_chiffres_MAJ.pdf"
    documents, metadatas = load_pdf(file_path)
    with open("../data/load_pdf.md", "w", encoding="utf-8") as f:
        f.write(f"# ======================================= DOCUMENTS ======================================= #\n"
                f"{'\n---\n'.join(documents)}\n"
                f"# ======================================= META ======================================= #"
                f"{'\n---\n'.join([json.dumps(meta, ensure_ascii=False) for meta in metadatas])}")


if __name__ == "__main__":
    main()
