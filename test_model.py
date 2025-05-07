from transformers import AutoTokenizer, T5ForConditionalGeneration

BASE_MODEL_ID = "google/flan-t5-base"
MODEL_PATH = "trained_model"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

test_question = "Cos'è l'ipoteca navale?"
inputs = tokenizer(test_question, return_tensors="pt")

outputs = model.generate(
    inputs.input_ids,
    max_new_tokens=250,  # Max tokens for the generated response
    num_beams=1,  # Use 1 for sampling, >1 for beam search
    temperature=0.7,  # Controls randomness (0.0 deterministic, >0 random)
    do_sample=True,  # Enable sampling if temperature > 0
    top_k=50,  # Sample from top K tokens
    top_p=0.95,  # Sample from top P cumulative probability
    # early_stopping=True # Can stop generation early if end-of-sequence token is generated
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Question: {test_question}")
print(f"Response: {response}")