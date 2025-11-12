import os
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import sacrebleu
import evaluate
from comet import download_model, load_from_checkpoint
import torch
from peft import PeftModel

# ========== File Paths ========== #
LOCATION = "/Users/foopanda/en-hi.txt/Ubuntu.en-hi"
SRC_FILE = f"{LOCATION}.en"
TGT_FILE =f"{LOCATION}.hi"

OUT_FILE = r"/Users/foopanda/predictions.hi"

# ========== Load mBART Model ========== #
model_name = "facebook/mbart-large-50-many-to-many-mmt"
src_lang_code = "en_XX"
tgt_lang_code = "hi_IN"
def get_tokenizer(model_name):
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    tokenizer.src_lang = src_lang_code
    return tokenizer
model = MBartForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ========== Read First 500 Lines ========== #
with open(SRC_FILE, "r", encoding="utf-8") as f:
    src_sentences = [line.strip() for line in f.readlines()]

with open(TGT_FILE, "r", encoding="utf-8") as f:
    refs = [line.strip() for line in f.readlines()]

test_size = 500
src_sentences = src_sentences[-1*test_size:]
refs = refs[-1*test_size:]


def translate_and_evaluate(model, tokenizer, src_sentences, refs, name):
    # ========== Translate ========== #
    preds = []
    batch_size = 8
    print(f"Translating using {name}...")

    for i in range(0, len(src_sentences), batch_size):
        batch = src_sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        outputs = model.generate(
            **inputs,
            max_length=512,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang_code]
        )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds.extend(decoded)
        #print(decoded)
        print(f"Translated {i + len(batch)}/{len(src_sentences)}")

    # ========== Save Predictions ========== #
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for line in preds:
            f.write(line + "\n")

    # ========== Evaluation ========== #
    print("\n=== Evaluation Results (English → Hindi, Last 500 Lines) ===")

    # BLEU
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    print(f"BLEU for {name}: {round(bleu.score, 2)}")

    # ROUGE-L
    rouge = evaluate.load("rouge")
    rouge_result = rouge.compute(predictions=preds, references=refs)
    print(f"ROUGE-L for {name}: {round(rouge_result['rougeL'], 4)}")

    # METEOR
    meteor = evaluate.load("meteor")
    meteor_result = meteor.compute(predictions=preds, references=refs)
    print(f"METEOR for {name}: {round(meteor_result['meteor'], 4)}")

    # CHRF
    chrf = evaluate.load("chrf")
    chrf_result = chrf.compute(predictions=preds, references=refs)
    print(f"CHRF for {name}: {round(chrf_result['score'], 4)}")


#translate_and_evaluate(model, get_tokenizer(model_name), src_sentences, refs, "original")
lora_adapter = './mbart-lora-finetuned/checkpoint-10000'
#lora_adapter = f"./mbart-ubuntu-{src_lang_code}-{tgt_lang_code}-adapter"
translate_and_evaluate(PeftModel.from_pretrained(model, lora_adapter), 
    get_tokenizer(lora_adapter), 
    src_sentences, refs, "fine-tuned")

# COMET
# print("Running COMET evaluation...")
# comet_model_path = download_model("Unbabel/wmt22-comet-da")
# comet_model = load_from_checkpoint(comet_model_path)
# comet_data = [{"src": s, "mt": t, "ref": r} for s, t, r in zip(src_sentences, preds, refs)]
# comet_result = comet_model.predict(comet_data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)

# comet_key = "system_score" if "system_score" in comet_result else "mean_score"
# print(f"COMET: {round(comet_result[comet_key], 4)}")
