# src/eval.py
import argparse, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.math_tools import extract_numeric_from_text, verify_numeric_answer
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--dataset', required=True)  # path to jsonl
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto')
device = next(model.parameters()).device

total = 0
correct = 0
with open(args.dataset, 'r', encoding='utf-8') as f:
    for line in f:
        total += 1
        obj = json.loads(line)
        q = obj.get('question')
        expected = obj.get('answer') or obj.get('solution') or ''
        prompt = f"Q: {q}\nA:"
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        out_text = tokenizer.decode(out[0], skip_special_tokens=True).split('A:')[-1].strip()
        ok, num = extract_numeric_from_text(out_text)
        exp_ok, exp_num = extract_numeric_from_text(expected)
        if ok and exp_ok:
            if abs(num - exp_num) < 1e-3:
                correct += 1
        else:
            # fallback: string match simplified (naive)
            if out_text.strip() == expected.strip():
                correct += 1
print(f"Total={total}, Correct={correct}, Acc={correct/total if total>0 else 0.0}")
