adapter_name = "WikiMatrix"
LOCATION = f"/Users/foopanda/en-hi.txt/{adapter_name}.en-hi"
SRC_FILE = f"{LOCATION}.en"
TGT_FILE = f"{LOCATION}.hi"
source_lang = "en_XX"
target_lang = "hi_IN"
total_lines = 5*16
batch_size = 16  # Adjust based on M1 memory (8-16 works well)

import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from functools import reduce
from tqdm import tqdm
import gc
from collections import defaultdict
import random

# Load model and tokenizer
model_name = "facebook/mbart-large-50-many-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

# Set source and target languages
tokenizer.src_lang = source_lang
tokenizer.tgt_lang = target_lang

# Enable MPS (Metal Performance Shaders) for M1
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

def model_name_predicate(name):
    return not("_norm" in name) and not("model.shared" in name) and not("embedding" in name)

def load_parallel_data(source_file, target_file, num_lines):
    """Load parallel text files and clean them"""
    with open(source_file, 'r', encoding='utf-8') as f:
        source_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    with open(target_file, 'r', encoding='utf-8') as f:
        target_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # Ensure same length
    random_index = random.sample(range(len(source_lines)), num_lines)
    source_line_final = [source_lines[i] for i in random_index]
    target_line_final = [target_lines[i] for i in random_index]
    
    return source_line_final, target_line_final

# Load all data
source_lines, target_lines = load_parallel_data(SRC_FILE, TGT_FILE, total_lines)
print(f"Loaded {len(source_lines)} parallel sentences")

# Initialize gradient accumulation dictionary
accumulated_grads = {}
for name, param in model.named_parameters():
    if param.requires_grad and model_name_predicate(name):
        accumulated_grads[name] = torch.zeros_like(param.data).cpu()

# Enable training mode
model.train()

# Process in batches
num_batches = (len(source_lines) + batch_size - 1) // batch_size
total_loss = 0.0

print(f"\nProcessing {num_batches} batches of size {batch_size}...")

for batch_idx in tqdm(range(num_batches)):
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(source_lines))
    
    batch_sources = source_lines[start_idx:end_idx]
    batch_targets = target_lines[start_idx:end_idx]
    
    # Tokenize batch
    inputs = tokenizer(
        batch_sources,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch_targets,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).input_ids
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = labels.to(device)
    
    # Forward pass
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    
    # Scale loss by batch size for proper averaging
    scaled_loss = loss / num_batches
    total_loss += loss.item()
    
    # Backward pass
    scaled_loss.backward()
    
    # Accumulate gradients
    for name, param in model.named_parameters():
        if model_name_predicate(name):
            if (param.grad is not None):
                accumulated_grads[name] += param.grad.data.cpu()
            else:
                print(f"{name} grad is None")
    
    # Clear gradients for next batch
    model.zero_grad()
    
    del inputs, labels, outputs, loss, scaled_loss
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
        torch.mps.synchronize()  # Ensure operations are complete

print(f"\nAverage Loss across all batches: {total_loss / num_batches:.4f}")

# Compute gradient norms
print("\nComputing gradient norms...")
grad_norms = []

module_grads = defaultdict(lambda: {'params': {}, 'total_norm': 0.0, 'total_params': 0})
module_names = []

for name, param in model.named_parameters():
    if name in accumulated_grads:
        avg_grad = accumulated_grads[name]
        param_size = reduce(lambda a, b: a * b, avg_grad.shape, 1)
        grad_norm_squared = (avg_grad.norm().item() ** 2) #* param_size
        
        if model_name_predicate(name):
            # Extract module name (remove .weight or .bias suffix)
            if name.endswith('.weight') or name.endswith('.bias'):
                module_name = name.rsplit('.', 1)[0]
            else:
                module_name = name
        
            # Accumulate for this module
            module_names.append(module_name)
            module_grads[module_name]['params'].update({name: avg_grad.shape})
            module_grads[module_name]['total_norm'] += grad_norm_squared
            module_grads[module_name]['total_params'] += param_size

# Sort by gradient norm (descending)
def get_gradient_norm_per_param(key):
    norm_sq = module_grads[key]['total_norm']
    count = module_grads[key]['total_params']
    return norm_sq/(count*count)

#print()
module_names.sort(key=get_gradient_norm_per_param, reverse=True)

# Print results and calculate LoRA parameters
print("\nGradient norms per parameter (sorted):")
print("-" * 100)
total_params = 0

lora_layers = []
one_dim_layers = []

for key in module_names:
    # Calculate trainable params for LoRA configuration
    if model_name_predicate(key):
        params = module_grads[key]['params']
        is_lora = False
        print(f"layer: {key} has gradient norm per parameter: {get_gradient_norm_per_param(key)}")
        for shape in params.values():
            if len(shape) == 2:  # Matrix layers (suitable for LoRA)
                # LoRA params = r * (m + n) where m, n are matrix dimensions
                # Assuming r=16: params ≈ 16 * (dim1 + dim2)
                lora_params = 16 * sum(shape)
                total_params += lora_params
                is_lora = True
            else:  # Bias/layer norm (train fully)
                total_params += module_grads[key]['total_params']
        if (is_lora):
            lora_layers.append(key)
        else:
            one_dim_layers.append(key)
    
    if total_params >= 5e6:
        print(f"\n--- Reached ~5M trainable parameters at this layer ---")
        break

print("lora layers to train:")
lora_layer_string = ", ".join([f'"{name}"' for name in lora_layers])
print(f'[{lora_layer_string}]')

print("\none dimensional layers to train:")
one_dim_layer_string = ", ".join([f'"{name}"' for name in one_dim_layers])
print(f'[{one_dim_layer_string}]')


print(f"\nTotal trainable parameters (with LoRA r=16): {total_params:,}")