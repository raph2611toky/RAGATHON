import os
import json
from tqdm import tqdm
import pdfplumber
import pytesseract
from PIL import Image
import cv2
import numpy as np

# ========================== CONFIGURATION ==========================
fichier_pdf = os.path.abspath("./data/MESUPRES_en_chiffres_MAJ.pdf")
dossier_sortie = os.path.abspath("./extracted_graphs")
os.makedirs(dossier_sortie, exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extraire_graphs(pdf_path, output_dir):
    graphs_info = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num in tqdm(range(len(pdf.pages))):
                page = pdf.pages[page_num]
                text = page.extract_text() or ""

                if "Graphe" in text:
                    lines = text.split('\n')
                    title = None
                    for i, line in enumerate(lines):
                        if line.strip().startswith("Graphe"):
                            title = line.strip()
                            if i + 1 < len(lines) and not lines[i+1].strip().startswith("Tableau"):
                                title += " " + lines[i+1].strip()
                            break

                    # Render page to image
                    page_image = page.to_image(resolution=300)
                    image_filename = f"page{page_num+1}_full.png"
                    image_path = os.path.join(output_dir, image_filename)
                    page_image.save(image_path, format="PNG")

                    # Use OpenCV to detect graph border
                    img_cv = cv2.imread(image_path)
                    if img_cv is None:
                        continue
                    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    graph_bbox = None
                    max_area = 0
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        area = w * h
                        aspect_ratio = w / float(h)
                        if area > max_area and w > 100 and h > 100 and 0.5 < aspect_ratio < 2:
                            max_area = area
                            graph_bbox = (x, y, w, h)

                    images_saved = [image_path]
                    ocr_results = []
                    cropped_path = None
                    if graph_bbox:
                        x, y, w, h = graph_bbox
                        cropped = img_cv[y:y+h, x:x+w]
                        cropped_filename = f"page{page_num+1}_graph_cropped.png"
                        cropped_path = os.path.join(output_dir, cropped_filename)
                        cv2.imwrite(cropped_path, cropped)
                        images_saved.append(cropped_path)

                    # OCR on cropped or full image
                    ocr_path = cropped_path if cropped_path else image_path
                    try:
                        img_pil = Image.open(ocr_path)
                        ocr_text = pytesseract.image_to_string(img_pil, lang='fra')
                        ocr_results = [ocr_text]
                    except Exception as e:
                        ocr_results = [f"OCR Error: {str(e)}"]

                    graphs_info.append({
                        "page": page_num + 1,
                        "title": title,
                        "text": text,
                        "images": images_saved,
                        "ocr_results": ocr_results,
                        "graph_bbox": graph_bbox
                    })
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")

    json_path = os.path.join(output_dir, "graphs_info.json")
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(graphs_info, json_file, ensure_ascii=False, indent=4)

    print(f"Extraction terminée. Infos sauvegardées dans {json_path}")

extraire_graphs(fichier_pdf, dossier_sortie)