# ========== File Paths ========== #
source_datatset = "WikiMatrix"
target_dataset = "WikiMatrix"
LOCATION = f"../Zip_training_data/en-hi.txt/{target_dataset}.en-hi"
checkpoint = None#"checkpoint-2000"
SRC_FILE = f"{LOCATION}.en"
TGT_FILE =f"{LOCATION}.hi"


import os
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import sacrebleu
import evaluate
from comet import download_model, load_from_checkpoint
import torch
import pytorch_lightning as ptl
import numpy as np
from peft import PeftModel
import warnings


# ========== Load mBART Model ========== #
model_name = "facebook/mbart-large-50-many-to-many-mmt"
src_lang_code = "en_XX"
tgt_lang_code = "hi_IN"

tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
tokenizer.src_lang = src_lang_code

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_base_model(model_name):
    model = MBartForConditionalGeneration.from_pretrained(model_name,
        torch_dtype=torch.float16,  # Use fp16 for efficiency
        device_map="auto")
    model = model.to(device)
    return model

# ========== Read First 500 Lines ========== #
with open(SRC_FILE, "r", encoding="utf-8") as f:
    src_sentences = [line.strip() for line in f.readlines()]

with open(TGT_FILE, "r", encoding="utf-8") as f:
    refs = [line.strip() for line in f.readlines()]

test_size = 504
src_sentences = src_sentences[:test_size]
refs = refs[:test_size]


def get_lora_model(model_name, adapter):
    lora_model = PeftModel.from_pretrained(get_base_model(model_name), lora_adapter)
    print(f"Before merge: {type(lora_model).__name__}")
    lora_model = lora_model.merge_and_unload()  # Merges LoRA into base model
    print(f"After merge: {type(lora_model).__name__}")
    lora_model.eval()
    return lora_model

import time 
def translate_and_evaluate(model, tokenizer, src_sentences, refs, name):
    OUT_FILE = f"../Zip_training_data/predictions_{name}.hi"
    preds = []
    batch_size = 8
    print(f"Translating using {name}...")
    
    total_generate_time = 0
    total_tokens_generated = 0

    for i in range(0, len(src_sentences), batch_size):
        batch = src_sentences[i:i+batch_size]
        inputs = tokenizer(batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128).to(device)


        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang_code]
            )
        generate_time = time.time() - start
        total_generate_time += generate_time
        
        # CHECK OUTPUT LENGTHS
        output_lengths = [len(seq) for seq in outputs]
        avg_length = sum(output_lengths) / len(output_lengths)
        total_tokens_generated += sum(output_lengths)
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds.extend(decoded)
        
        print(f"Batch {i//batch_size + 1}: generate={generate_time:.2f}s, avg_tokens={avg_length:.1f}, max_tokens={max(output_lengths)}")
        
        # PRINT FIRST FEW TRANSLATIONS TO SEE IF THEY MAKE SENSE
        if i == 0:
            print("\nSample translations from first batch:")
            for j in range(min(2, len(decoded))):
                print(f"  SRC: {batch[j][:100]}...")
                print(f"  TGT: {decoded[j][:100]}...")
                print(f"  Tokens: {output_lengths[j]}")
    
    print(f"\nTotal tokens generated: {total_tokens_generated}")
    print(f"Avg tokens per sentence: {total_tokens_generated / len(src_sentences):.1f}")
    # ... rest of code
    
    print(f"\n=== Timing Summary for {name} ===")
    print(f"Total generation: {total_generate_time:.2f}s")
    
    # Save predictions
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for line in preds:
            f.write(line + "\n")
    
    # Evaluation
    print("\n=== Evaluation Results ===")
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    print(f"BLEU for {name}: {round(bleu.score, 2)}")
    
    rouge = evaluate.load("rouge")
    rouge_result = rouge.compute(predictions=preds, references=refs)
    print(f"ROUGE-L for {name}: {round(rouge_result['rougeL'], 4)}")
    
    meteor = evaluate.load("meteor")
    meteor_result = meteor.compute(predictions=preds, references=refs)
    print(f"METEOR for {name}: {round(meteor_result['meteor'], 4)}")
    
    chrf = evaluate.load("chrf")
    chrf_result = chrf.compute(predictions=preds, references=refs)
    print(f"CHRF for {name}: {round(chrf_result['score'], 4)}")


#translate_and_evaluate(get_base_model(model_name), tokenizer, src_sentences, refs, "original")
lora_adapter = f"./mbart-lora-{source_datatset}-{src_lang_code}-{tgt_lang_code}" if checkpoint is None else f"./mbart-lora-{source_datatset}-{src_lang_code}-{tgt_lang_code}/{checkpoint}"
translate_and_evaluate(get_lora_model(model_name, lora_adapter), tokenizer, src_sentences, refs, "fine-tuned")

