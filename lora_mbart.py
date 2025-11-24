adapter_name = "WikiMatrix"
LOCATION = f"../Zip_training_data/en-hi.txt/{adapter_name}.en-hi"
SRC_FILE = f"{LOCATION}.en"
TGT_FILE =f"{LOCATION}.hi"
source_lang = "en_XX"
target_lang = "hi_IN"
dataset_cap = 20000
valid_size = 504

warmup_percent = 0.05
num_saves = 6

import torch
import re
from math import sqrt
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset, DatasetDict

adapter_location = None# f"./mbart-lora-{adapter_name}-{source_lang}-{target_lang}/checkpoint-2000"

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
    min_length = min(dataset_cap+valid_size, min(len(source_lines), len(target_lines)))
    source_lines = source_lines[:min_length]
    target_lines = target_lines[:min_length]
    
    return source_lines, target_lines

def create_dataset(source_file, target_file):
    """Create train/validation/test splits"""
    source, target = load_parallel_data(source_file, target_file)

    # Create dataset dictionary
    dataset_dict = DatasetDict({
        'train': Dataset.from_dict({
            'source': source[valid_size:],
            'target': target[valid_size:]
        }),
        'validation': Dataset.from_dict({
            'source': source[:valid_size],
            'target': target[:valid_size]
        })
    })
    
    return dataset_dict


def get_lora_config(decoder_finetune, grad_based_layer):
    target_regex = [
            # Decoder self-attention (target fluency)
            r"decoder\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)",
            
            # Cross-attention (source-target alignment)
            r"decoder\.layers\.\d+\.encoder_attn\.(q_proj|k_proj|v_proj|out_proj)",
            r"decoder\.layers.*.(fc1|fc2)"
        ]
    decode_modules = set()
    grad_based_layers = {"model.decoder.layers.0.self_attn.v_proj", "model.decoder.layers.0.self_attn.v_proj", "model.decoder.layers.0.self_attn.out_proj", "model.decoder.layers.0.self_attn.out_proj", "model.decoder.layers.11.self_attn.k_proj", "model.decoder.layers.11.self_attn.k_proj", "model.decoder.layers.2.self_attn.v_proj", "model.decoder.layers.2.self_attn.v_proj", "model.decoder.layers.5.self_attn.k_proj", "model.decoder.layers.5.self_attn.k_proj", "model.decoder.layers.4.self_attn.v_proj", "model.decoder.layers.4.self_attn.v_proj", "model.decoder.layers.1.self_attn.v_proj", "model.decoder.layers.1.self_attn.v_proj", "model.decoder.layers.0.encoder_attn.v_proj", "model.decoder.layers.0.encoder_attn.v_proj", "model.decoder.layers.5.self_attn.v_proj", "model.decoder.layers.5.self_attn.v_proj", "model.decoder.layers.5.self_attn.q_proj", "model.decoder.layers.5.self_attn.q_proj", "model.decoder.layers.11.fc1", "model.decoder.layers.11.fc1", "model.decoder.layers.3.self_attn.v_proj", "model.decoder.layers.3.self_attn.v_proj", "model.decoder.layers.0.encoder_attn.out_proj", "model.decoder.layers.0.encoder_attn.out_proj", "model.decoder.layers.1.self_attn.out_proj", "model.decoder.layers.1.self_attn.out_proj", "model.decoder.layers.2.self_attn.k_proj", "model.decoder.layers.2.self_attn.k_proj", "model.decoder.layers.0.fc2", "model.decoder.layers.0.fc2", "model.decoder.layers.11.self_attn.v_proj", "model.decoder.layers.11.self_attn.v_proj", "model.decoder.layers.2.self_attn.out_proj", "model.decoder.layers.2.self_attn.out_proj", "model.decoder.layers.6.self_attn.v_proj", "model.decoder.layers.6.self_attn.v_proj", "model.decoder.layers.2.self_attn.q_proj", "model.decoder.layers.2.self_attn.q_proj", "model.decoder.layers.0.encoder_attn.q_proj", "model.decoder.layers.0.encoder_attn.q_proj", "model.decoder.layers.1.encoder_attn.v_proj", "model.decoder.layers.1.encoder_attn.v_proj", "model.decoder.layers.5.self_attn.out_proj", "model.decoder.layers.5.self_attn.out_proj", "model.decoder.layers.7.self_attn.v_proj", "model.decoder.layers.7.self_attn.v_proj", "model.decoder.layers.7.self_attn.k_proj", "model.decoder.layers.7.self_attn.k_proj", "model.decoder.layers.4.self_attn.out_proj", "model.decoder.layers.4.self_attn.out_proj", "model.decoder.layers.0.self_attn.q_proj", "model.decoder.layers.0.self_attn.q_proj", "model.decoder.layers.8.self_attn.k_proj", "model.decoder.layers.8.self_attn.k_proj", "model.decoder.layers.8.self_attn.q_proj", "model.decoder.layers.8.self_attn.q_proj", "model.decoder.layers.9.self_attn.v_proj", "model.decoder.layers.9.self_attn.v_proj", "model.decoder.layers.3.self_attn.q_proj", "model.decoder.layers.3.self_attn.q_proj", "model.decoder.layers.2.encoder_attn.v_proj", "model.decoder.layers.2.encoder_attn.v_proj", "model.decoder.layers.0.fc1", "model.decoder.layers.0.fc1", "model.decoder.layers.3.self_attn.k_proj", "model.decoder.layers.3.self_attn.k_proj", "model.decoder.layers.7.self_attn.q_proj", "model.decoder.layers.7.self_attn.q_proj", "model.decoder.layers.1.fc2", "model.decoder.layers.1.fc2", "model.decoder.layers.8.self_attn.v_proj", "model.decoder.layers.8.self_attn.v_proj", "model.decoder.layers.9.self_attn.out_proj", "model.decoder.layers.9.self_attn.out_proj", "model.decoder.layers.1.encoder_attn.q_proj", "model.decoder.layers.1.encoder_attn.q_proj", "model.decoder.layers.0.encoder_attn.k_proj", "model.decoder.layers.0.encoder_attn.k_proj", "model.decoder.layers.1.encoder_attn.out_proj", "model.decoder.layers.1.encoder_attn.out_proj", "model.decoder.layers.3.encoder_attn.v_proj", "model.decoder.layers.3.encoder_attn.v_proj", "model.decoder.layers.0.self_attn.k_proj", "model.decoder.layers.0.self_attn.k_proj", "model.decoder.layers.11.fc2", "model.decoder.layers.11.fc2", "model.decoder.layers.6.self_attn.out_proj", "model.decoder.layers.6.self_attn.out_proj", "model.decoder.layers.6.self_attn.k_proj", "model.decoder.layers.6.self_attn.k_proj", "model.decoder.layers.3.self_attn.out_proj", "model.decoder.layers.3.self_attn.out_proj", "model.decoder.layers.4.self_attn.q_proj", "model.decoder.layers.4.self_attn.q_proj", "model.decoder.layers.1.fc1", "model.decoder.layers.1.fc1", "model.decoder.layers.4.encoder_attn.v_proj", "model.decoder.layers.4.encoder_attn.v_proj", "model.decoder.layers.11.self_attn.out_proj", "model.decoder.layers.11.self_attn.out_proj", "model.decoder.layers.6.self_attn.q_proj", "model.decoder.layers.6.self_attn.q_proj", "model.decoder.layers.7.self_attn.out_proj", "model.decoder.layers.7.self_attn.out_proj", "model.decoder.layers.1.self_attn.k_proj", "model.decoder.layers.1.self_attn.k_proj", "model.decoder.layers.4.fc1", "model.decoder.layers.4.fc1", "model.decoder.layers.2.fc2", "model.decoder.layers.2.fc2", "model.decoder.layers.3.fc2", "model.decoder.layers.3.fc2", "model.decoder.layers.10.self_attn.v_proj", "model.decoder.layers.10.self_attn.v_proj", "model.decoder.layers.10.self_attn.out_proj", "model.decoder.layers.10.self_attn.out_proj", "model.decoder.layers.4.fc2", "model.decoder.layers.4.fc2"}
    import re

    for name, module in base_model.named_modules():
        for pattern in target_regex:
            if re.search(pattern, name):
                print(name)
                decode_modules.add(name)
                break

    print(f"layers dropped for grad based layer: {decode_modules-grad_based_layers}")
    print(f"non decoder layers picked up based on grad: {grad_based_layers-decode_modules}")
    if (grad_based_layer):
        return LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=list(grad_based_layers),
            lora_dropout=0.05,
            bias="all",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
    elif (decoder_finetune):
        return LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=list(decode_modules),
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
    else:
        return LoraConfig(
            r=16,  # Rank of the low-rank matrices
            lora_alpha=32,  # Scaling factor
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj", 
                            "fc1", "fc2"],  # Which layers to apply LoRA to
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )


def get_fresh_tunable_model():
    # 1. Load model and tokenizer
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

    lora_config = get_lora_config(True, False)

    # 3. Apply LoRA to the model
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()  # Shows how many params are trainable
    #return base_model, tokenizer
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
        #padding="max_length"
    )
    
    # Set target language for labels
    tokenizer.tgt_lang = target_lang
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target"],
            max_length=128,
            truncation=True,
            #padding="max_length"
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# for name, param in model.named_parameters():
#     #if any(ln in name for ln in ['layer_norm', 'final_layer_norm', 'self_attn_layer_norm']):
#     if name in {"model.decoder.layers.11.self_attn_layer_norm", "model.decoder.layers.11.self_attn_layer_norm", "model.decoder.layers.0.final_layer_norm", "model.decoder.layers.0.final_layer_norm", "model.encoder.layer_norm", "model.encoder.layer_norm", "model.decoder.layers.11.final_layer_norm", "model.decoder.layers.11.final_layer_norm", "model.decoder.layernorm_embedding", "model.decoder.layernorm_embedding"}:
#         param.requires_grad = True
#         print(f"Unfrozen: {name}")

# 5. Training arguments
batch_size = 4
gradient_accumulation_steps = 1
effective_batch_size = gradient_accumulation_steps*batch_size
lr_multiplier = sqrt(effective_batch_size)
learning_rate = (5e-5)*lr_multiplier
num_epochs = 3#int(3*lr_multiplier/2)
total_steps = (num_epochs*dataset_cap)/effective_batch_size
warmup_steps = int(warmup_percent*total_steps)
save_steps = int(total_steps/num_saves)
eval_steps = int(total_steps/10)
logging_steps = 100

print(f"""Training with {num_epochs} epochs, 
    warmup steps of {warmup_steps}, 
    saving every {save_steps} steps, 
    with learning_rate {learning_rate} 
    and logging every {logging_steps} steps....""")
training_args = Seq2SeqTrainingArguments(
    output_dir=f"./mbart-lora-{adapter_name}-{source_lang}-{target_lang}",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    warmup_steps=warmup_steps,
    weight_decay=0.01,
    logging_steps=logging_steps,
    eval_steps=eval_steps,
    eval_strategy="steps",
    save_steps=save_steps,
    save_safetensors=False,
    save_total_limit=3,
    predict_with_generate=True,
    push_to_hub=False
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
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 8. Train the model
trainer.train()

# 9. Save the LoRA adapters
model.save_pretrained(f"./mbart-lora-{adapter_name}-{source_lang}-{target_lang}")
tokenizer.save_pretrained(f"./mbart-lora-{adapter_name}-{source_lang}-{target_lang}")