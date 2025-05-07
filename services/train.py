import os
import sys
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, DatasetDict

# Load tokenizer and base model
MODEL_ID = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

# Apply LoRA configuration
# r=16 is a good balance for a moderately sized adapter
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,  # Often set equal to r
    lora_dropout=0.05,
    task_type=TaskType.SEQ_2_SEQ_LM,
    target_modules=["q", "v"],  # Standard target modules for T5
)
model = get_peft_model(model, lora_config)

# Print the trainable parameters to confirm LoRA is applied
model.print_trainable_parameters()

# Load your data
# Ensure the path to your dataset is correct
DATA_PATH = "../data"
DATASET_PATH = os.path.join(DATA_PATH, "navy_trade_data.jsonl")
if not os.path.exists(DATASET_PATH):
    print(f"Error: Dataset file not found at {DATASET_PATH}")
    print("Please ensure your dataset is in the correct location.")
    sys.exit()

dataset = load_dataset("json", data_files=DATASET_PATH)

# Split the dataset into train and validation (e.g., 80/20)
# Using a fixed seed ensures the split is the same each time
train_test_split = dataset["train"].train_test_split(test_size=0.2, seed=42)
dataset = DatasetDict(
    {"train": train_test_split["train"], "validation": train_test_split["test"]}
)


# Tokenize your examples - Function handles batches
def tokenize_seq2seq(example):
    prompts = []
    targets = []

    for i in range(len(example["instruction"])):
        # Construct the prompt including instruction and optional input (context)
        prompt = (
            "Rispondi alla seguente domanda basata sul contesto:\n"
            + example["instruction"][i]
        )
        if example["input"][i]:  # Add input only if not empty
            prompt += "\nContesto: " + example["input"][i]
        prompts.append(prompt)

        # Target is the desired output
        targets.append(example["output"][i])

    # Tokenize the prompts (model inputs)
    # truncation=True and max_length limit sequence length
    # padding is handled by DataCollator
    input_tokens = tokenizer(prompts, truncation=True, max_length=512)

    # Tokenize the targets (labels for the model)
    target_tokens = tokenizer(targets, truncation=True, max_length=512)

    # Prepare the dictionary expected by the Trainer
    model_inputs = {
        "input_ids": input_tokens["input_ids"],
        "attention_mask": input_tokens["attention_mask"],
        # Labels are the input_ids of the target. DataCollator handles padding to -100.
        "labels": target_tokens["input_ids"],
    }

    return model_inputs


# Apply the tokenization function to the dataset splits
# batched=True processes examples in batches, which is more efficient
tokenized_dataset = dataset.map(tokenize_seq2seq, batched=True)

# Use a Data Collator for dynamic padding and label masking (-100)
# This is essential for efficient batch processing with variable sequence lengths
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Check for BF16 support and set precision accordingly
use_bf16 = (
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
)  # BF16 requires Ampere or newer GPU (compute capability >= 8.0)
use_fp16 = (
    torch.cuda.is_available() and not use_bf16
)  # Use FP16 if GPU is available but doesn't support BF16

# Training configuration
training_args = TrainingArguments(
    output_dir="../trained_model",  # Directory to save checkpoints
    per_device_train_batch_size=4,  # Increased batch size slightly if memory allows
    gradient_accumulation_steps=2,  # Accumulate gradients (n) to simulate a batch size of 4*2=8
    num_train_epochs=30,  # Number of training epochs. Monitor eval_loss for overfitting.
    fp16=use_fp16,  # Use FP16 if supported and BF16 is not
    bf16=use_bf16,  # Use BF16 if supported (recommended over FP16 if available)
    logging_steps=10,  # Log metrics every 10 steps
    save_strategy="epoch",  # Save a checkpoint at the end of each epoch
    learning_rate=2e-4,  # Learning rate for the optimizer
    eval_strategy="epoch",  # Evaluate at the end of each epoch
    logging_dir="./logs",  # Directory for training logs (useful with TensorBoard)
    load_best_model_at_end=True,  # Load the model with the best eval_loss at the end of training
    metric_for_best_model="eval_loss",  # Metric to determine the best model
    greater_is_better=False,  # For loss, lower is better
    weight_decay=0.01,  # Add weight decay to help prevent overfitting
    save_total_limit=3,  # Limit the total number of checkpoints saved
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],  # Specify the training dataset split
    eval_dataset=tokenized_dataset[
        "validation"
    ],  # Specify the validation dataset split
    data_collator=data_collator,  # Use the data collator
)

# Start training
print("Starting training...")
trainer.train()
print("Training completed. Saving model...")

# Save the fine-tuned LoRA weights (Trainer might have already saved the best)
# This ensures the adapter is saved in the specified directory
model.save_pretrained("../trained_model")

# Save the tokenizer as well for easy loading later
tokenizer.save_pretrained("../trained_model")

print("Model and tokenizer saved to ./trained_model")
