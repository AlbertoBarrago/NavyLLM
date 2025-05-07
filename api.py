import json
import os
import torch
import faiss
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

# --- Configurazione ---
# Modello LLM
BASE_MODEL_ID = "google/flan-t5-small"
MODEL_PATH = "./trained_model"

# RAG
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
FAISS_INDEX_FILE = "data/maritime_dataset_rag.faiss"
CHUNKS_FILE = "data/maritime_dataset_rag_chunks.json"
NUM_RETRIEVED_CHUNKS = 3

# Creazione dell'app FastAPI
app = FastAPI(
    title="Navy LLM API",
    description="API RAG-LLM per interrogare il modello della Marina",
    version="1.0.0",
)

# Abilita CORS per consentire richieste da frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Modello di dati per la richiesta
class QuestionRequest(BaseModel):
    question: str


# Modello di dati per la risposta
class AnswerResponse(BaseModel):
    answer: str
    retrieved_contexts: List[str]


# --- Caricamento modelli e indici ---
print(f"Caricamento del modello LLM da: {MODEL_PATH}")
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = T5ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        low_cpu_mem_usage=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"Modello LLM caricato e spostato su {device}.")
except RuntimeError as e:
    print(f"Errore durante il caricamento del modello LLM: {e}")
    raise HTTPException(status_code=500, detail=f"Errore nel caricamento del modello: {str(e)}")

print(f"Caricamento del modello di embedding: {EMBEDDING_MODEL_NAME}")
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Modello di embedding caricato.")
except RuntimeError as e:
    print(f"Errore durante il caricamento del modello di embedding: {e}")
    raise HTTPException(status_code=500, detail=f"Errore nel caricamento del modello di embedding: {str(e)}")

print("Caricamento dei componenti RAG (indice FAISS e chunks)...")
try:
    if not os.path.exists(FAISS_INDEX_FILE):
        raise FileNotFoundError(f"File di indice FAISS non trovato: {FAISS_INDEX_FILE}")
    faiss_index = faiss.read_index(FAISS_INDEX_FILE)
    print(f"Indice FAISS caricato con {faiss_index.ntotal} vettori.")

    if not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError(f"File dei chunks di testo non trovato: {CHUNKS_FILE}")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        text_chunks = json.load(f)
    print(f"Caricati {len(text_chunks)} chunks di testo.")
    print("Componenti RAG caricati con successo.")
except Exception as e:
    print(f"Errore durante il caricamento dei componenti RAG: {e}")
    raise HTTPException(status_code=500, detail=f"Errore nel caricamento dei componenti RAG: {str(e)}")


# --- Funzioni di supporto ---
def retrieve_context(
        query: str,
        index: faiss.Index,
        chunks: list,
        embedding_model_response: SentenceTransformer,
        num_results: int,
) -> list:
    """Recupera i chunk di testo più rilevanti per una query."""
    query_embedding = embedding_model_response.encode(query)
    query_embedding = np.array([query_embedding]).astype("float32")
    distances, indices = index.search(query_embedding, num_results)
    retrieved_texts = [chunks[i] for i in indices[0]]
    return retrieved_texts


def generate_response_with_context(
        query: str,
        context: list,
        model_response: T5ForConditionalGeneration,
        tokenizer_response: AutoTokenizer,
        device_response: torch.device,
) -> str:
    """Genera una risposta utilizzando il modello LLM."""
    prompt = "Rispondi alla seguente domanda basata sul contesto fornito:\n"
    prompt += "Contesto: " + "\n".join(context) + "\n"
    prompt += "Domanda: " + query + "\n"

    inputs = tokenizer_response(
        prompt, return_tensors="pt", max_length=512, truncation=True
    ).to(device_response)

    with torch.no_grad():
        outputs = model_response.generate(
            inputs.input_ids,
            max_new_tokens=250,
            num_beams=1,
            temperature=0.6,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )

    response = tokenizer_response.decode(outputs[0], skip_special_tokens=True)
    return response


# --- Endpoint API ---
@app.get("/")
def read_root():
    """Endpoint principale che fornisce informazioni di base sull'API."""
    return {
        "name": "Navy LLM API",
        "description": "API per interrogare il modello LLM della Marina",
        "usage": "Invia una richiesta POST a /ask con un JSON contenente una 'question'"
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Endpoint per porre domande al modello LLM.

    Invia una domanda in italiano e ottieni una risposta generata
    dal modello RAG-LLM utilizzando il contesto più rilevante.
    """
    query = request.question.strip()

    if not query:
        raise HTTPException(status_code=400, detail="La domanda non può essere vuota")

    # Fase di recupero
    retrieved_context = retrieve_context(
        query, faiss_index, text_chunks, embedding_model, NUM_RETRIEVED_CHUNKS
    )

    if not retrieved_context:
        return {
            "answer": "Mi dispiace, non ho trovato informazioni rilevanti per rispondere alla tua domanda.",
            "retrieved_contexts": []
        }

    # Fase di generazione
    response = generate_response_with_context(
        query, retrieved_context, model, tokenizer, device
    )

    return {
        "answer": response,
        "retrieved_contexts": retrieved_context
    }


# --- Per esecuzione con uvicorn ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)