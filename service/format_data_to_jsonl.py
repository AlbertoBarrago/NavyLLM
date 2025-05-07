import json
import os

input_text_file = "data/extracted_codice_navigazione.txt"
output_jsonl_file = "data/navy_trade_data.jsonl"

os.makedirs("data", exist_ok=True)

with open(input_text_file, 'r', encoding='utf-8') as f:
    content = f.read()

pages = content.split("\n--- Pagina Fine ---\n")

examples = []
for i, page in enumerate(pages):
    if page.strip():
        example = {
            "instruction": f"Estrai informazioni rilevanti dalla seguente pagina del Codice della Navigazione:",
            "input": page.strip(),
            "output": f"Questa è la pagina {i+1} del Codice della Navigazione. Contiene normative relative alla navigazione marittima."
        }
        examples.append(example)

with open(output_jsonl_file, 'w', encoding='utf-8') as f:
    for example in examples:
        f.write(json.dumps(example, ensure_ascii=False) + '\n')

print(f"Convertito in formato JSONL. Creati {len(examples)} esempi in '{output_jsonl_file}'")