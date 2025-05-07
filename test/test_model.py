from transformers import AutoTokenizer, T5ForConditionalGeneration

BASE_MODEL_ID = "google/flan-t5-base"
MODEL_PATH = "./trained_model"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

# Test with a question from your dataset
test_question = "Qual è la differenza tra 'porto franco' e 'deposito doganale'?"
inputs = tokenizer(test_question, return_tensors="pt")

outputs = model.generate(
    inputs.input_ids,
    max_new_tokens=150,
    temperature=0.7,
    do_sample=True,
    top_k=50,
    top_p=0.95
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Question: {test_question}")
print(f"Response: {response}")