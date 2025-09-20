import os
import json
from tqdm import tqdm
import pdfplumber
import pytesseract
from PIL import Image
import cv2
import numpy as np
import re

# ========================== CONFIGURATION ==========================
fichier_pdf = os.path.abspath("../data/MESUPRES_en_chiffres_MAJ.pdf")
dossier_sortie = os.path.abspath("../extracted_graphs")
os.makedirs(dossier_sortie, exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = r'C:\Programmes\tesseract-OCR\tesseract.exe'

def extraire_graphs(pdf_path, output_dir):
    graphs_dict = {}
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num in tqdm(range(len(pdf.pages))):
                page = pdf.pages[page_num]
                text = page.extract_text() or ""

                if "Graphe" in text:
                    lines = text.split('\n')
                    titles = []
                    for i, line in enumerate(lines):
                        if line.strip().startswith("Graphe"):
                            title = line.strip()
                            if i + 1 < len(lines) and not lines[i+1].strip().startswith("Tableau"):
                                title += " " + lines[i+1].strip()
                            titles.append(title)

                    # Render page to image for visual detection
                    page_image = page.to_image(resolution=300)
                    image_filename = f"temp_page{page_num+1}_full.png"
                    temp_image_path = os.path.join(output_dir, image_filename)
                    page_image.save(temp_image_path, format="PNG")

                    # Use OpenCV to detect potential graph borders
                    img_cv = cv2.imread(temp_image_path)
                    if img_cv is None:
                        continue
                    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Get table bboxes to exclude
                    tables = page.find_tables()
                    table_bboxes = [table.bbox for table in tables]  # (x0, y0, x1, y1)

                    graph_candidates = []
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        area = w * h
                        aspect_ratio = w / float(h)
                        if area > 10000 and w > 200 and h > 200 and 0.5 < aspect_ratio < 2:  # Adjust thresholds for graphs
                            candidate_bbox = (x, y, x + w, y + h)  # (x0, y0, x1, y1)
                            is_table = False
                            for tb in table_bboxes:
                                # Check overlap; if significant overlap, assume it's a table
                                overlap_x = max(0, min(candidate_bbox[2], tb[2]) - max(candidate_bbox[0], tb[0]))
                                overlap_y = max(0, min(candidate_bbox[3], tb[3]) - max(candidate_bbox[1], tb[1]))
                                overlap_area = overlap_x * overlap_y
                                if overlap_area > area * 0.5:  # >50% overlap
                                    is_table = True
                                    break
                            if not is_table:
                                graph_candidates.append((x, y, w, h))

                    # Sort candidates by position (top-to-bottom, left-to-right)
                    graph_candidates.sort(key=lambda bbox: (bbox[1], bbox[0]))  # by y, then x

                    for idx, graph_bbox in enumerate(graph_candidates):
                        x, y, w, h = graph_bbox
                        cropped = img_cv[y:y+h, x:x+w]
                        cropped_filename = f"page{page_num+1}_graph{idx}.png"
                        cropped_path = os.path.join(output_dir, cropped_filename)
                        cv2.imwrite(cropped_path, cropped)

                        try:
                            # os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/tessdata/'
                            img_pil = Image.open(cropped_path)
                            ocr_text = pytesseract.image_to_string(img_pil, lang='fra')
                        except Exception as e:
                            ocr_text = f"OCR Error: {str(e)}"

                        title = titles[idx] if idx < len(titles) else None

                        if title:
                            match = re.search(r"Graphe (\d+)", title, re.IGNORECASE)
                            if match:
                                graph_num = match.group(1)
                                key = f"graph {graph_num}"
                                graphs_dict[key] = cropped_path
                            else:
                                os.remove(cropped_path)
                        else:
                            os.remove(cropped_path)

                    os.remove(temp_image_path)

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")

    json_path = os.path.join(output_dir, "graphs_mapping.json")
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(graphs_dict, json_file, ensure_ascii=False, indent=4)

    print(f"Extraction terminée. Infos sauvegardées dans {json_path}")

extraire_graphs(fichier_pdf, dossier_sortie)