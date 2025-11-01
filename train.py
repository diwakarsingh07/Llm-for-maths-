# src/train.py
import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch

def prepare_dataset(tokenizer, ds_path, max_length=512):
    import json
    records = []
    with open(ds_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            prompt = f"Q: {obj.get('question')}\nA:"
            # simple target is solution or answer
            target = obj.get('solution') or obj.get('answer') or ''
            text = prompt + " " + target
            records.append({'text': text})
    from datasets import Dataset
    d = Dataset.from_list(records)
    def tokenize_fn(ex):
        return tokenizer(ex['text'], truncation=True, max_length=max_length)
    return d.map(tokenize_fn, batched=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--train_file', required=True)
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--use_lora', action='store_true')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    ds = prepare_dataset(tokenizer, args.train_file)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        save_strategy='epoch',
        logging_steps=10,
        fp16=torch.cuda.is_available()
    )

    # For simplicity we use Trainer (PEFT/LoRA omitted in this small example)
    def data_collator(features):
        import torch
        input_ids = [f['input_ids'] for f in features]
        attention_mask = [f['attention_mask'] for f in features]
        input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in input_ids], batch_first=True, padding_value=tokenizer.pad_token_id or 0)
        attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in attention_mask], batch_first=True, padding_value=0)
        labels = input_ids.clone()
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    trainer = Trainer(model=model, args=training_args, train_dataset=ds, data_collator=data_collator)
    trainer.train()
    trainer.save_model(args.output_dir)
    print('Model saved to', args.output_dir)

if __name__ == '__main__':
    main()
