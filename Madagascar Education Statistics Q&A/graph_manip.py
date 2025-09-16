import os
import json
from tqdm import tqdm
import fitz  # from pymupdf
import pytesseract
from PIL import Image

# ========================== CONFIGURATION ==========================
fichier_pdf = os.path.abspath("./data/MESUPRES_en_chiffres_MAJ.pdf")
dossier_sortie = os.path.abspath("./extracted_graphs")
os.makedirs(dossier_sortie, exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\tesseract\tesseract-ocr-w64-setup-5.5.0.20241111.exe'

def extraire_graphs(pdf_path, output_dir):
    pdf = fitz.open(pdf_path)
    graphs_info = []

    for page_num in tqdm(range(len(pdf))):
        page = pdf[page_num]
        text = page.get_text("text")

        if "Graphe" in text:
            lines = text.split('\n')
            title = None
            for i, line in enumerate(lines):
                if line.strip().startswith("Graphe"):
                    title = line.strip()
                    if i + 1 < len(lines) and not lines[i+1].strip().startswith("Tableau"):
                        title += " " + lines[i+1].strip()
                    break

            image_list = page.get_images(full=True)
            images_saved = []
            ocr_results = []
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = f"page{page_num+1}_graph{img_index}.{image_ext}"
                image_path = os.path.join(output_dir, image_filename)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                images_saved.append(image_path)

                # OCR on image for labels/values (basic, may need tuning)
                img_pil = Image.open(image_path)
                ocr_text = pytesseract.image_to_string(img_pil)
                ocr_results.append(ocr_text)

            graphs_info.append({
                "page": page_num + 1,
                "title": title,
                "text": text,
                "images": images_saved,
                "ocr_results": ocr_results  # Extracted text from images (potential labels/values)
            })

    pdf.close()

    json_path = os.path.join(output_dir, "graphs_info.json")
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(graphs_info, json_file, ensure_ascii=False, indent=4)

    print(f"Extraction terminée. Infos sauvegardées dans {json_path}")

extraire_graphs(fichier_pdf, dossier_sortie)