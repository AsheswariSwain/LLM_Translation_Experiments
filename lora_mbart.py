adapter_name = "WikiMatrix"
LOCATION = f"/Users/foopanda/en-hi.txt/{adapter_name}.en-hi"
SRC_FILE = f"{LOCATION}.en"
TGT_FILE =f"{LOCATION}.hi"
source_lang = "en_XX"
target_lang = "hi_IN"

import torch
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset, DatasetDict

adapter_location = None

model_name = "facebook/mbart-large-50-many-to-many-mmt"
base_model = MBartForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use fp16 for efficiency
        device_map="auto"
    )

def load_parallel_data(source_file, target_file):
    """Load parallel text files and clean them"""
    with open(source_file, 'r', encoding='utf-8') as f:
        source_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    with open(target_file, 'r', encoding='utf-8') as f:
        target_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # Ensure same length
    min_length = min(len(source_lines), len(target_lines))
    source_lines = source_lines[:min_length]
    target_lines = target_lines[:min_length]
    
    return source_lines, target_lines

def create_dataset(source_file, target_file, train_split=0.95):
    """Create train/validation/test splits"""
    source, target = load_parallel_data(source_file, target_file)
    
    total = len(source)
    train_size = int(total * train_split)
    
    # Create dataset dictionary
    dataset_dict = DatasetDict({
        'train': Dataset.from_dict({
            'source': source[:train_size],
            'target': target[:train_size]
        }),
        'validation': Dataset.from_dict({
            'source': source[train_size:],
            'target': target[train_size:]
        })
    })
    
    return dataset_dict


def get_fresh_tunable_model():
    # 1. Load model and tokenizer
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

    # 2. Configure LoRA
    lora_config = LoraConfig(
        r=16,  # Rank of the low-rank matrices
        lora_alpha=32,  # Scaling factor
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", 
                        "fc1", "fc2"],  # Which layers to apply LoRA to
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    # 3. Apply LoRA to the model
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()  # Shows how many params are trainable
    return model, tokenizer

def get_preexisting_tunable_model(adapter_location):
    # 1. Load model and tokenizer
    model = PeftModel.from_pretrained(base_model, adapter_location, is_trainable=True)
    tokenizer = MBart50TokenizerFast.from_pretrained(adapter_location)
    model.print_trainable_parameters()  # Shows how many params are trainable
    return model, tokenizer

if (adapter_location is None):
    model, tokenizer = get_fresh_tunable_model()
else:
    model, tokenizer = get_preexisting_tunable_model(adapter_location)

# 4. Prepare your dataset
# Example: translation task
dataset = create_dataset(SRC_FILE, TGT_FILE) 

# Set source and target languages
tokenizer.src_lang = source_lang  # Source language code
tokenizer.tgt_lang = target_lang  # Target language code

def preprocess_function(examples):
    #tokenizer.src_lang = source_lang
    
    # Tokenize inputs (English)
    model_inputs = tokenizer(
        examples["source"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    
    # Set target language for labels
    #tokenizer.tgt_lang = target_lang
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target"],
            max_length=128,
            truncation=True,
            padding="max_length"
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# 5. Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./mbart-lora-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    #per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
    eval_steps=500,
    save_steps=1000,
    save_total_limit=3,
    predict_with_generate=True,
    push_to_hub=False,
)

# 6. Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# 7. Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    #eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 8. Train the model
trainer.train()

# 9. Save the LoRA adapters
model.save_pretrained(f"./mbart-{adapter_name}-{source_lang}-{target_lang}-adapter")
tokenizer.save_pretrained(f"./mbart-{adapter_name}-{source_lang}-{target_lang}-adapter")