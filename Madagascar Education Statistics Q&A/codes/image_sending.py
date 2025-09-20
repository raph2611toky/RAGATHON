import google.generativeai as genai
from PIL import Image
from typing import List, Tuple, Dict
import regex
import re
import json
import time

def get_gemini_response_with_images(query: str, contexts: List[Tuple[str, Dict]], image_paths: List[str] = []) -> Dict:
    # 1. Préparer le contenu de la requête, y compris le texte et les images.
    prompt_parts = [make_rag_prompt(query, contexts)]

    # 2. Charger les images et les ajouter au prompt.
    for path in image_paths:
        try:
            img = Image.open(path)
            prompt_parts.append(img)
        except Exception as e:
            print(f"Erreur lors du chargement de l'image {path}: {e}")
            continue

    # 3. Configurer et appeler le modèle comme d'habitude.
    api_keys = ["YOUR_API_KEY_1", "YOUR_API_KEY_2"] # Remplacez par vos clés API
    model = genai.GenerativeModel("gemini-1.5-flash")
    max_retries = 3
    retry_delay = 5

    for key_index, api_key in enumerate(api_keys):
        try:
            genai.configure(api_key=api_key)
            for attempt in range(max_retries):
                try:
                    res = model.generate_content(prompt_parts) # Envoyer la liste de parties
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
                            if "answer" in response_data:
                                response_data["answer"] = str(response_data["answer"]) if response_data["answer"] is not None else "Aucune réponse claire"
                            if "doc_index" in response_data:
                                response_data["doc_index"] = str(response_data["doc_index"]) if response_data["doc_index"] is not None else "0"
                        except json.JSONDecodeError as e:
                            answer_match = re.search(r'answer:?\s*([^\n]+)', response_text, re.IGNORECASE)
                            context_match = re.search(r'"doc_index":?\s*([^\n]+)', response_text, re.IGNORECASE)
                            response_data = {
                                "answer": str(answer_match.group(1).strip()) if answer_match else "Aucune réponse claire",
                                "doc_index": str(context_match.group(1).strip()) if context_match else "0"
                            }
                    else:
                        answer_match = re.search(r'answer:?\s*([^\n]+)', response_text, re.IGNORECASE)
                        context_match = re.search(r'doc_index:?\s*([^\n]+)', response_text, re.IGNORECASE)
                        response_data = {
                            "answer": str(answer_match.group(1).strip()) if answer_match else "Aucune réponse claire",
                            "doc_index": str(context_match.group(1).strip()) if context_match else "0"
                        }
                    # ... (reste du code pour le traitement de la réponse)
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
    return {"answer": "Erreur : Toutes les clés API ont dépassé leur quota ou ont échoué", "doc_index": "0"}

# Exemple d'utilisation
# response = get_gemini_response_with_images(
#     query="Quel est le contenu de l'image ?",
#     contexts=[],
#     image_paths=["path/to/your/image.png"]
# )