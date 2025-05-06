from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, DatasetDict

MODEL_ID = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type=TaskType.SEQ_2_SEQ_LM,
    target_modules=["q", "v"]
)
model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

dataset = load_dataset("json", data_files="navy_trade_data.jsonl")

train_test_split = dataset["train"].train_test_split(test_size=0.2, seed=42)
dataset = DatasetDict({
    'train': train_test_split['train'],
    'validation': train_test_split['test']
})


def tokenize_seq2seq(example):
    prompts = []
    targets = []

    for i in range(len(example["instruction"])):
        prompt = "Rispondi alla seguente domanda basata sul contesto:\n" + example["instruction"][i]
        if example["input"][i]:
            prompt += "\nContesto: " + example["input"][i]
        prompts.append(prompt)

        targets.append(example["output"][i])

    input_tokens = tokenizer(prompts, truncation=True, max_length=512)

    target_tokens = tokenizer(targets, truncation=True, max_length=512)

    model_inputs = {
        "input_ids": input_tokens["input_ids"],
        "attention_mask": input_tokens["attention_mask"],
        "labels": target_tokens["input_ids"]
    }

    return model_inputs

tokenized_dataset = dataset.map(tokenize_seq2seq, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = TrainingArguments(
    output_dir="./trained_model",
    per_device_train_batch_size=2,
    num_train_epochs=30,
    fp16=False,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    eval_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
)

print("Inizio addestramento...")
trainer.train()
print("Addestramento completato. Salvataggio del modello...")

model.save_pretrained("./trained_model")

tokenizer.save_pretrained("./trained_model")

print("Modello e tokenizer salvati in ./trained_model")
