import os
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Union
from pypdf import PdfReader
from tqdm import tqdm
import chromadb
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
import pandas as pd
import csv
import google.generativeai as genai

# Load environment variables
load_dotenv()

def load_pdf(file_path: str) -> Tuple[List[str], List[Dict]]:
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

def get_embedding_function() -> GoogleGenerativeAiEmbeddingFunction:
    """
    Creates a Google Generative AI embedding function using the API key.

    Returns:
        GoogleGenerativeAiEmbeddingFunction: Configured embedding function.
    """
    google_api_key = os.getenv("GEMINI_API_KEY")
    if not google_api_key:
        google_api_key = input("Please enter your Google API Key: ")
        os.environ["GEMINI_API_KEY"] = google_api_key

    return GoogleGenerativeAiEmbeddingFunction(api_key=google_api_key)

def load_data(documents: List[str], metadatas: List[Dict], collection_name: str) -> None:
    """
    Loads documents and their embeddings into ChromaDB.

    Args:
        documents (List[str]): List of document texts to load.
        metadatas (List[Dict]): List of metadata for each document.
        collection_name (str): Name of the ChromaDB collection.
    """
    client = chromadb.EphemeralClient()
    embedding_function = get_embedding_function()

    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_function
    )

    count = collection.count()
    print(f"Collection already contains {count} documents")
    ids = [str(i) for i in range(count, count + len(documents))]

    # Load documents in batches of 100
    for i in tqdm(
        range(0, len(documents), 100), desc="Adding documents", unit_scale=100
    ):
        collection.add(
            ids=ids[i:i + 100],
            documents=documents[i:i + 100],
            metadatas=metadatas[i:i + 100]
        )
    print("Documents loaded successfully")

def get_db(collection_name: str) -> chromadb.Collection:
    """
    Retrieves a ChromaDB collection instance.

    Args:
        collection_name (str): Name of the collection to retrieve.

    Returns:
        chromadb.Collection: The ChromaDB collection.
    """
    client = chromadb.EphemeralClient()
    embedding_function = get_embedding_function()
    
    return client.get_collection(
        name=collection_name, embedding_function=embedding_function
    )

def get_relevant_passage(query: str, db: chromadb.Collection, n_results: int = 1) -> Tuple[List[str], List[Dict]]:
    """
    Queries the ChromaDB collection for relevant passages.

    Args:
        query (str): The query text.
        db (chromadb.Collection): The ChromaDB collection.
        n_results (int): Number of results to return.

    Returns:
        Tuple[List[str], List[Dict]]: Relevant documents and their metadata.
    """
    results = db.query(
        query_texts=[query], n_results=n_results, include=["documents", "metadatas"]
    )
    return results["documents"][0], results["metadatas"][0]

def make_rag_prompt(query: str, relevant_passages: Union[str, List[str]]) -> str:
    """
    Builds a RAG prompt from query and relevant passages.

    Args:
        query (str): User query.
        relevant_passages (Union[str, List[str]]): Single passage or list of passages.

    Returns:
        str: Formatted RAG prompt.
    """
    if isinstance(relevant_passages, str):
        relevant_passages = [relevant_passages]

    escaped_passages = [
        passage.replace("'", "").replace('"', "").replace("\n", " ")
        for passage in relevant_passages
    ]
    combined_passages = "\n---\n".join(escaped_passages)

    return f"""You are a helpful and informative bot that answers questions using text from the reference passages below. 
Your response must be direct, no preamble or irrelevant phrases.  
If the passages are irrelevant to the answer, you may ignore them.

QUESTION: '{query}'
PASSAGES:
{combined_passages}

ANSWER:
"""

def get_gemini_response(query: str, context: List[str]) -> str:
    """
    Queries the Gemini API to generate a response.

    Args:
        query (str): The user query.
        context (List[str]): Relevant context passages.

    Returns:
        str: The generated response.
    """
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    prompt = make_rag_prompt(query, context)
    response = model.generate_content(prompt)
    
    return response.text

def process_questions_from_csv(db: chromadb.Collection, csv_path: str, output_csv: str = "../submissions/submission_file.csv") -> None:
    """
    Reads questions from a CSV file, generates answers using RAG, and saves results to a CSV.

    Args:
        db (chromadb.Collection): ChromaDB collection instance.
        csv_path (str): Path to input CSV with questions.
        output_csv (str): Path to output CSV for results.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    results = []

    for _, row in df.iterrows():
        qid = row["id"]
        question = row["question"]
        
        print(f"Answering question {qid}: {question}")
        
        relevant_texts, metadatas = get_relevant_passage(question, db, n_results=1)
        answer = get_gemini_response(question, relevant_texts)
        print(f"Answer: {answer}")
        print("========================================================\n")

        results.append({
            "id": qid,
            "question": question,
            "answer": answer,
            "context": relevant_texts,
            "ref_page": metadatas[0]["page_number"]
        })

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "answer", "context", "ref_page"])
        writer.writeheader()
        for row in results:
            writer.writerow({
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "context": str(row["context"]),
                "ref_page": str(row["ref_page"])
            })

    print(f"✅ Results written to {output_csv}")

# Main execution
def main():
    # Load PDF and extract data
    file_path = "../data/MESUPRES_en_chiffres_MAJ.pdf"
    collection_name = "rag"
    
    documents, metadatas = load_pdf(file_path)
    load_data(documents, metadatas, collection_name)
    
    db = get_db(collection_name)
    process_questions_from_csv(db, "../data/questions.csv")

if __name__ == "__main__":
    main()