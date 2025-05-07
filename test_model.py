import torch
import faiss
import numpy as np
import json
import os
from transformers import AutoTokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer

BASE_MODEL_ID = "google/flan-t5-base"
MODEL_PATH = "./trained_model"

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
FAISS_INDEX_FILE = "maritime_rag.faiss"
CHUNKS_FILE = "maritime_rag_chunks.json"
NUM_RETRIEVED_CHUNKS = 3

print(f"Loading LLM model from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"LLM model loaded and moved to {device}.")

print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Embedding model loaded.")

print(f"Loading FAISS index from: {FAISS_INDEX_FILE}")
if not os.path.exists(FAISS_INDEX_FILE):
    print(f"Error: FAISS index file not found: {FAISS_INDEX_FILE}")
    print("Please run build_rag_index.py first to create the index.")
    exit()
faiss_index = faiss.read_index(FAISS_INDEX_FILE)
print(f"FAISS index loaded with {faiss_index.ntotal} vectors.")

print(f"Loading text chunks from: {CHUNKS_FILE}")
if not os.path.exists(CHUNKS_FILE):
     print(f"Error: Text chunks file not found: {CHUNKS_FILE}")
     print("Please run build_rag_index.py first to create the chunks.")
     exit()
with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
    text_chunks = json.load(f)
print(f"Loaded {len(text_chunks)} text chunks.")

def retrieve_context(query, index, chunks, embedding_model, num_results):
    """
    Retrieves the most relevant text chunks for a given query from the FAISS index.

    Args:
        query (str): The user's query.
        index (faiss.Index): The loaded FAISS index.
        chunks (list): A list of original text chunks corresponding to the index.
        embedding_model (SentenceTransformer): The model used to generate embeddings.
        num_results (int): The number of top relevant chunks to retrieve.

    Returns:
        list: A list of strings, where each string is a retrieved text chunk.
    """
    query_embedding = embedding_model.encode(query)
    query_embedding = np.array([query_embedding]).astype('float32')

    distances, indices = index.search(query_embedding, num_results)

    retrieved_texts = [chunks[i] for i in indices[0]]

    return retrieved_texts

def generate_response_with_context(query, context, model, tokenizer, device):
    """
    Generates a response using the LLM model, the original query, and the retrieved context.

    Args:
        query (str): The user's original query.
        context (list): A list of retrieved text chunks to be used as context.
        model (T5ForConditionalGeneration): The fine-tuned T5 model.
        tokenizer (AutoTokenizer): The tokenizer for the T5 model.
        device (torch.device): The device (CPU or GPU) the model is on.

    Returns:
        str: The generated response from the LLM.
    """
    prompt = "Rispondi alla seguente domanda basata sul contesto fornito:\n"
    prompt += "Contesto: " + "\n".join(context) + "\n"
    prompt += "Domanda: " + query + "\n"

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)

    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=250,
        num_beams=1,
        temperature=0.8,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

if __name__ == "__main__":
    print("\n--- Query the RAG Model ---")
    print("Type your question or 'esci' to exit.")

    while True:
        user_query = input("\nYour question: ").strip()

        if user_query.lower() == 'esci':
            break

        if not user_query:
            print("Please enter a valid question.")
            continue

        print("Retrieving context...")
        retrieved_context = retrieve_context(user_query, faiss_index, text_chunks, embedding_model, NUM_RETRIEVED_CHUNKS)

        print("\nRetrieved Context:")
        if retrieved_context:
            for i, chunk in enumerate(retrieved_context):
                print(f"--- Chunk {i+1} ---")
                print(chunk)
                print("-" * 10)
        else:
            print("No relevant context found.")

        print("\nGenerating response...")
        rag_response = generate_response_with_context(user_query, retrieved_context, model, tokenizer, device)

        print("\nResponse from RAG Model:")
        print(rag_response)
        print("-" * 30)

    print("Session ended.")
