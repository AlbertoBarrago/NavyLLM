import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, T5ForConditionalGeneration
import os
from contextlib import asynccontextmanager # Import asynccontextmanager

# --- Configuration ---
# LLM Model (Generator) Configuration
BASE_MODEL_ID = "google/flan-t5-base"
MODEL_PATH = "./trained_model" # Directory where your fine-tuned T5 model is saved

# Global variables to hold the loaded model
tokenizer = None
model = None
device = None

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the tokenizer and LLM model when the FastAPI application starts.
    This function runs before the app starts serving requests.
    """
    global tokenizer, model, device

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Tokenizer and LLM Model
    print(f"Loading LLM model from: {MODEL_PATH}")
    try:
        # Load the tokenizer from the base model
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        # Load the fine-tuned model, potentially with LoRA weights
        model = T5ForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            # Adjust torch_dtype based on your training (fp16/bf16 or float32)
            # If trained with fp16/bf16, loading with torch.float32 might be slower
            # or require more memory on GPU, but is generally safe.
            # If trained with float32, keep torch.float32.
            # You can remove this line if your model was saved in default precision.
            # torch_dtype=torch.float32,
            low_cpu_mem_usage=True # Helps with memory usage on CPU during loading
        )
        model.to(device).eval() # Move model to device and set to evaluation mode
        print("LLM model loaded successfully.")
    except Exception as e:
        print(f"Error loading LLM model: {e}")
        # Raise an exception to prevent the app from starting if the model can't load
        raise RuntimeError(f"Could not load LLM model: {e}")

    # The `yield` statement is where the application starts serving requests.
    # Code after yield runs on shutdown.
    yield

    # --- Shutdown Event: Clean up resources (Optional but recommended) ---
    # For this simple case, explicit cleanup might not be strictly necessary
    # as Python's garbage collector handles it, but it's good practice for
    # more complex resources like database connections.
    print("Shutting down application.")
    # Example cleanup (though models usually don't need explicit closing):
    # del model
    # del tokenizer
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()


# --- FastAPI App Initialization (with lifespan) ---
app = FastAPI(lifespan=lifespan) # Pass the lifespan function here

# --- Pydantic Model for Request Body ---
class Query(BaseModel):
    question: str

# --- FastAPI Endpoint ---
@app.post("/ask")
async def ask(query: Query):
    """
    Receives a question and generates a response using the T5 model directly (No RAG).
    """
    if model is None or tokenizer is None:
        # This check should ideally not be needed if lifespan loaded correctly,
        # but it's a safe fallback.
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        # Construct the prompt using only the user's question
        # Use the same prompt format as your training data for instructions without context
        prompt = "Rispondi alla seguente domanda basata sul contesto:\n" + query.question
        # If your training data *always* had a context prefix even when input was empty,
        # you might need to adjust the prompt format here.
        # Example if you always used "Contesto: \nDomanda:..." format:
        # prompt = "Rispondi alla seguente domanda basata sul contesto:\nContesto: \nDomanda: " + query.question


        # Tokenize the prompt and move to the device
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)

        # Generate the response using the T5 model
        with torch.no_grad(): # Disable gradient calculation for inference
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=250, # Max tokens for the generated response
                num_beams=1, # Use 1 for sampling, >1 for beam search
                temperature=0.8, # Controls randomness (0.0 deterministic, >0 random)
                do_sample=True, # Enable sampling if temperature > 0
                top_k=50, # Sample from top K tokens
                top_p=0.95, # Sample from top P cumulative probability
                # early_stopping=True # Can stop generation early if end-of-sequence token is generated
            )

        # Decode the generated output
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"question": query.question, "response": response}

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")

# --- Example Usage (if running directly, typically you use uvicorn) ---
# To run this application:
# 1. Save the code as main.py (or another name).
# 2. Ensure train.py was run and created the ./trained_model directory.
# 3. Install uvicorn: pip install uvicorn
# 4. Run from your terminal in the project root directory:
#    uvicorn main:app --reload
# 5. You can then send POST requests to http://127.0.0.1:8000/ask
#    Example using curl:
#    curl -X POST -H "Content-Type: application/json" -d '{"question": "Cos\'è il deposito doganale?"}' http://1.2.3.4:8000/ask
    # Replace 1.2.3.4 with your server IP if not running locally
