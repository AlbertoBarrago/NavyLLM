import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

RAW_TEXT_DATA_FILE = "../extracted_codice_navigazione_for_rag.json"

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

FAISS_INDEX_FILE = "../maritime_rag.faiss"
CHUNKS_FILE = "../maritime_rag_chunks.json"

def load_raw_text_data(filename):
    if not os.path.exists(filename):
        print(f"Errore: File '{filename}' non trovato: {filename}")
        return []
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def chunk_text(text, chunk_size, chunk_overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks

def build_faiss_index(chunks, model):
    print(f"Generando embedding per {len(chunks)} chunk...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    print("Generazione embedding completata.")

    embeddings = np.array(embeddings).astype('float32')

    embedding_dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(embedding_dim)

    print(f"Aggiungendo {len(embeddings)} vettori all'indice FAISS...")
    index.add(embeddings)
    print(f"Indice FAISS costruito con {index.ntotal} vettori.")

    return index

if __name__ == "__main__":
    raw_data = load_raw_text_data(RAW_TEXT_DATA_FILE)
    if not raw_data:
        print("Nessun dato da processare. Assicurati che il file RAW_TEXT_DATA_FILE esista e contenga dati.")
    else:
        full_text = ""
        for entry in raw_data:
            full_text += entry.get("text", "") + "\n--- Documento Fine ---\n"

        text_chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"Testo diviso in {len(text_chunks)} chunk.")

        print(f"Caricando modello di embedding: {EMBEDDING_MODEL_NAME}")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Modello di embedding caricato.")

        faiss_index = build_faiss_index(text_chunks, embedding_model)

        print(f"Salvando indice FAISS in '{FAISS_INDEX_FILE}'...")
        faiss.write_index(faiss_index, FAISS_INDEX_FILE)
        print("Indice FAISS salvato.")

        print(f"Salvando chunk in '{CHUNKS_FILE}'...")
        with open(CHUNKS_FILE, 'w', encoding='utf-8') as f:
            json.dump(text_chunks, f, ensure_ascii=False, indent=4)
        print("Chunk salvati.")

        print("\nProcesso di costruzione indice completato.")
        print("Ora puoi usare il file test_model.py (modificato per RAG) per interrogare il modello.")
