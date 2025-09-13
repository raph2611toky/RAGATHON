import google.generativeai as genai
import os, re
import warnings
from dotenv import load_dotenv, find_dotenv

#============================Config============================#
warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv(find_dotenv())
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    os.environ["GEMINI_API_KEY"] = api_key
print(api_key)

def make_rag_prompt(query: str, context: str) -> str:
    return f"""
    Expert en stats éducatives Madagascar. Répondez en français, basé sur le contexte. Soyez précis, concis, factuel. Pas d'info hors contexte. Pour nombres/pourcentages, donnez exactement comme dans le contexte.

    **Question**: {query}
    **Contexte**: {context}
    **Réponse**:
    """
    
def extract_num(ans: str) -> str:
    match = re.search(r'\b\d+\b|[\d.]+%|[\d,]+\b', ans)
    return match.group(0) if match else ans


def get_gemini_response(query: str, context: str) -> str:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    q_type = "generale"
    prompt = make_rag_prompt(query, context)
    res = genai.GenerativeModel("gemini-2.5-flash-lite").generate_content(prompt)
    return extract_num(res.text) if q_type in ["num", "pct", "success"] else res.text

print(get_gemini_response("quelle est la superficies de madagascar?", "tu es un IA très intelligent."))