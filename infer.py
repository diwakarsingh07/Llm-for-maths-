# src/infer.py
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--question', required=True)
parser.add_argument('--max_new_tokens', type=int, default=256)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto')
device = next(model.parameters()).device

prompt = f"Q: {args.question}\nA:"
inputs = tokenizer(prompt, return_tensors='pt').to(device)
out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
text = tokenizer.decode(out[0], skip_special_tokens=True)
# print only generated answer portion
print(text.split('A:')[-1].strip())
