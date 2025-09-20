import os
from dotenv import load_dotenv, find_dotenv
from typing import List, Tuple, Dict, Optional
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

# Comment out Gemini API loading if not needed for this script
# load_dotenv(find_dotenv())
# api_keys = [
#     os.getenv("GEMINI_API_KEY"),
#     os.getenv("GEMINI_API_KEY_SECOND"),
# ]
# api_keys = [key for key in api_keys if key]
# if not api_keys:
#     raise ValueError("No valid Gemini API keys found in environment variables")

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