import sys
import json
import os
import torch
import faiss
import numpy as np

from transformers import AutoTokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer

# --- Configuration ---
# LLM Model (Generator) Configuration
# ID of the base T5 model (used to load the correct tokenizer)
BASE_MODEL_ID = "google/flan-t5-small"
# Directory where your fine-tuned T5 model (with LoRA weights) is saved
MODEL_PATH = "trained_model"

# RAG (Retrieval) Configuration
# Name of the embedding model to use (must match the one used in build_rag_index.py)
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
# File of the FAISS index saved by build_rag_index.py (using the JSONL dataset)
FAISS_INDEX_FILE = "data/maritime_dataset_rag.faiss"
# File of the original text chunks saved by build_rag_index.py
CHUNKS_FILE = "data/maritime_dataset_rag_chunks.json"
# Number of top relevant chunks to retrieve for each query. Experiment with this value.
NUM_RETRIEVED_CHUNKS = 3

# --- Load Models and Index ---

# Load the tokenizer and the fine-tuned LLM model
print(f"Loading LLM model from: {MODEL_PATH}")
try:
    # Load the tokenizer from the base model to ensure compatibility
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    # Load the T5 model with the fine-tuned (LoRA) weights
    model = T5ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        low_cpu_mem_usage=True,  # Helps with memory usage during loading on CPU
    )
    # Determine and move the model to the appropriate device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()  # Set the model to evaluation mode
    print(f"LLM model loaded and moved to {device}.")
except RuntimeError as e:
    print(f"Error loading LLM model: {e}")
    sys.exit()  # Exit if the LLM model cannot be loaded

# Load the embedding model, used for creating vector representations of the text
print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    # embedding_model.to(device) # Uncomment to move the embedding model to GPU if preferred
    print("Embedding model loaded.")
except RuntimeError as e:
    print(f"Error loading embedding model: {e}")
    sys.exit()  # Exit if the embedding model cannot be loaded

# Load the FAISS index and the corresponding text chunks
print("Loading RAG components (FAISS index and chunks)...")
try:
    # Check if the FAISS index file exists before attempting to load
    if not os.path.exists(FAISS_INDEX_FILE):
        raise FileNotFoundError(f"FAISS index file not found: {FAISS_INDEX_FILE}")
    faiss_index = faiss.read_index(FAISS_INDEX_FILE)
    print(f"FAISS index loaded with {faiss_index.ntotal} vectors.")

    # Check if the text chunks file exists before attempting to load
    if not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError(f"Text chunks file not found: {CHUNKS_FILE}")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        text_chunks = json.load(f)
    print(f"Loaded {len(text_chunks)} text chunks.")
    print("RAG components loaded successfully.")

except FileNotFoundError as e:
    print(f"Error loading RAG components: {e}")
    print(
        "Please ensure build_rag_index_from_jsonl.py was run successfully and the files exist."
    )
    sys.exit()  # Exit if RAG components cannot be loaded
except ValueError as e:
    print(f"An unexpected error occurred loading RAG components: {e}")
    sys.exit()  # Exit on other loading errors


# --- Helper Function for Retrieval ---
def retrieve_context(
    query: str,
    index: faiss.Index,
    chunks: list,
    embedding_model_response: SentenceTransformer,
    num_results: int,
) -> list:
    """
    Retrieves the most relevant text chunks for a given query from the FAISS index.

    Args:
        query (str): The user's query.
        index (faiss.Index): The loaded FAISS index.
        chunks (list): A list of original text chunks corresponding to the index.
        embedding_model_response (SentenceTransformer): The model used to generate embeddings.
        num_results (int): The number of top relevant chunks to retrieve.

    Returns:
        list: A list of strings, where each string is a retrieved text chunk.
    """
    # Create embedding of the query using the same model used for chunk embeddings
    query_embedding = embedding_model_response.encode(query)
    # Ensure the embedding is in float32 format and has the correct shape for FAISS
    query_embedding = np.array([query_embedding]).astype("float32")

    # Search the FAISS index for the most similar vectors to the query embedding
    # D: distances, I: indices of the nearest chunks
    distances, indices = index.search(query_embedding, num_results)

    # Retrieve the original text chunks corresponding to the found indices
    retrieved_texts = [chunks[i] for i in indices[0]]

    return retrieved_texts


# --- Helper Function for Generation with Context ---
def generate_response_with_context(
    query: str,
    context: list,
    model_response: T5ForConditionalGeneration,
    tokenizer_response: AutoTokenizer,
    device_response: torch.device,
) -> str:
    """
    Generates a response using the LLM model, the original query, and the retrieved context.

    Args:
        query (str): The user's original query.
        context (list): A list of retrieved text chunks to be used as context.
        model_response (T5ForConditionalGeneration): The fine-tuned T5 model.
        tokenizer_response (AutoTokenizer): The tokenizer for the T5 model.
        device_response (torch.device): The device (CPU or GPU) the model is on.

    Returns:
        str: The generated response from the LLM.
    """
    # Construct the prompt for the T5 model.
    # This format should match the one used in your train.py's tokenize_seq2seq function
    # to teach the model how to use the context.
    prompt = "Rispondi alla seguente domanda basata sul contesto fornito:\n"
    # Join the retrieved chunks into a single context block within the prompt
    prompt += "Contesto: " + "\n".join(context) + "\n"
    # Add the user's original question
    prompt += "Domanda: " + query + "\n"
    # The model is trained to generate the response after the question

    # Tokenize the complete prompt and move it to the appropriate device
    # max_length should be enough to contain the query and context
    inputs = tokenizer_response(
        prompt, return_tensors="pt", max_length=512, truncation=True
    ).to(device_response)

    # Generate the response using the T5 model
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model_response.generate(
            inputs.input_ids,
            max_new_tokens=250,  # Maximum number of tokens to generate for the response
            num_beams=1,  # Use 1 for sampling, >1 for beam search (1 is often better for RAG)
            temperature=0.6,  # Controls randomness (0.0 for deterministic, >0 for random)
            do_sample=True,  # Enable sampling if temperature > 0
            top_k=50,
            top_p=0.95,
        )

    # Decode the generated output tokens into a human-readable string
    response = tokenizer_response.decode(outputs[0], skip_special_tokens=True)

    return response


# --- Interactive Test Loop ---
if __name__ == "__main__":
    print("\n--- Query the RAG-LLM System ---")
    print("Type your question in Italian or 'esci' to exit.")

    while True:
        user_query = input("\nLa tua domanda: ").strip()

        if user_query.lower() == "esci":
            break

        if not user_query:
            print("Please enter a valid question.")
            continue

        # --- 1. Retrieval Phase ---
        print("Retrieving context...")
        # Call the retrieve_context function to find relevant chunks from the indexed dataset
        retrieved_context = retrieve_context(
            user_query, faiss_index, text_chunks, embedding_model, NUM_RETRIEVED_CHUNKS
        )

        # Print the retrieved context (optional but useful for debugging and understanding)
        print("\nRetrieved Context:")
        if retrieved_context:
            for i, chunk in enumerate(retrieved_context):
                print(f"--- Chunk {i + 1} ---")
                print(chunk)
                print("-" * 10)
        else:
            print("No relevant context found in the dataset.")

        # --- 2. Generation Phase ---
        print("\nGenerating response...")
        # Call the generate_response_with_context function to get the answer from the LLM
        # Pass the original query and the retrieved context
        rag_response = generate_response_with_context(
            user_query, retrieved_context, model, tokenizer, device
        )

        # Print the final response from the RAG-LLM system
        print("\nResponse from RAG-LLM:")
        print(rag_response)
        print("-" * 30)

    print("Session ended.")
