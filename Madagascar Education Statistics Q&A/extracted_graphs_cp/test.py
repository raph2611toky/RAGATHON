import cv2
import pytesseract
from pathlib import Path

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img_path = Path("page28_graph3.png")
img = cv2.imread(str(img_path))

h, w = img.shape[:2]
crop = img[0:int(h*0.3), :]

text = pytesseract.image_to_string(crop, lang="fra+eng", config="--oem 1 --psm 6")
print("Texte OCR détecté :")
print(text)
