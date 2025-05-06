import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, T5ForConditionalGeneration

app = FastAPI()
BASE_MODEL_ID = "google/flan-t5-base"  # Original base model
MODEL_PATH = "./trained_model"  # Path to your fine-tuned model

# Use base model tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

# Load the full model directly - since you saved the full model, not just adapters
model = T5ForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)
model = model.to("cpu").eval()
print("Model loaded successfully")

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    """
    Processes a query to generate a response using the T5 model.
    """
    # Format input - simplify to match your training format
    prompt = query.question

    # Create input tensors
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=150,
            do_sample=True,
            top_p=0.95,
            temperature=0.7
        )

    # Decode the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"response": response}