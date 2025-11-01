# data/prepare_dataset.py
# Simple utility to prepare small JSONL dataset files for training.
import argparse, json
from datasets import load_dataset

def write_jsonl(records, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['gsm8k'])
    parser.add_argument('--out', default='data/processed')
    args = parser.parse_args()

    out = args.out
    import os
    os.makedirs(out, exist_ok=True)

    records = []
    for ds in args.datasets:
        if ds.lower() == 'gsm8k':
            d = load_dataset('gsm8k', 'main', split='train')
            for item in d:
                # gsm8k fields: question, answer (latex), etc.
                q = item.get('question') or item.get('problem') or ''
                a = item.get('answer') or ''
                records.append({'question': q.strip(), 'solution': a.strip(), 'answer': a.strip()})
        else:
            print('Unknown dataset:', ds)

    train_path = os.path.join(out, 'train.jsonl')
    write_jsonl(records, train_path)
    print('Wrote', train_path, 'records:', len(records))

if __name__ == '__main__':
    main()
