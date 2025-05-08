import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

DATA_PATH = "data"
DATASET_FILE = os.path.join(
    DATA_PATH, "navy_trade_data.jsonl"
)

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
FAISS_INDEX_FILE = os.path.join(DATA_PATH, "maritime_dataset_rag.faiss")
CHUNKS_FILE = os.path.join(DATA_PATH, "maritime_dataset_rag_chunks.json")


def load_dataset_data(filename):
    """
    Loads data from a JSONL file.

    Args:
        filename (str): The path to the JSONL file.

    Returns:
        list: A list of dictionaries, where each dictionary is a parsed JSON object from a line.
              Returns an empty list if the file is not found or empty.
    """
    if not os.path.exists(filename):
        print(f"Error: Dataset file not found: {filename}")
        return []
    data = []
    with open(filename, "r", encoding="utf-8") as handle_file:
        for line in handle_file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line: {line.strip()} - {e}")
                continue
    return data


def create_rag_chunks_from_dataset(dataset_entries):
    """
    Creates text chunks for RAG indexing from dataset entries.
    Each chunk combines the 'input' and 'output' fields of an entry.

    Args:
        dataset_entries (list): A list of dictionaries loaded from the JSONL dataset.

    Returns:
        list: A list of strings, where each string is a text chunk representing
              the combined input and output of a dataset entry.
    """
    rag_chunks_from_dataset = []
    
    for entry in dataset_entries:
        chunk_text = ""
        if entry.get("input"):
            chunk_text += "Contesto: " + entry["input"] + "\n"
        if entry.get("output"):
            chunk_text += "Risposta: " + entry["output"]

        # Add the chunk only if it contains significant text
        if chunk_text.strip():
            rag_chunks_from_dataset.append(chunk_text.strip())

    # Note: This function assumes each dataset entry (input + output)
    # is a suitable size for a RAG chunk. If input/output fields are
    # extremely long, you might need to add further chunking logic here.
    return rag_chunks_from_dataset


def build_faiss_index(chunks, model):
    """
    Generates embeddings for text chunks and builds a FAISS index.

    Args:
        chunks (list): A list of text strings (the RAG chunks).
        model (SentenceTransformer): The embedding model used to encode the chunks.

    Returns:
        faiss.Index: The built FAISS index containing the chunk embeddings.
    """
    print(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    print("Embedding generation completed.")

    embeddings = np.array(embeddings).astype("float32")

    embedding_dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(embedding_dim)

    print(f"Adding {len(embeddings)} vectors to the FAISS index...")
    index.add(embeddings)  # pylint: disable=no-value-for-parameter
    print(f"FAISS index built with {index.ntotal} vectors.")

    return index


if __name__ == "__main__":
    dataset_data = load_dataset_data(DATASET_FILE)
    if not dataset_data:
        print(
            "No data to process. Ensure the DATASET_FILE exists and contains valid JSONL data."
        )
    else:
        rag_chunks = create_rag_chunks_from_dataset(dataset_data)
        print(f"Created {len(rag_chunks)} RAG chunks from the dataset.")

        if not rag_chunks:
            print(
                "No valid chunks created from the dataset. Check the content of the JSONL file."
            )
        else:
            print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print("Embedding model loaded.")

            faiss_index = build_faiss_index(rag_chunks, embedding_model)

            print(f"Saving FAISS index to '{FAISS_INDEX_FILE}'...")
            faiss.write_index(faiss_index, FAISS_INDEX_FILE)
            print("FAISS index saved.")

            print(f"Saving chunks to '{CHUNKS_FILE}'...")
            with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
                json.dump(rag_chunks, f, ensure_ascii=False, indent=4)
            print("Chunks saved.")

            print("\nIndex building process completed.")
            print(
                f"Index and chunks saved as '{FAISS_INDEX_FILE}' and '{CHUNKS_FILE}'."
            )
            print(
                "Now you can use the test_model.py (RAG version) pointing to these new files."
            )
