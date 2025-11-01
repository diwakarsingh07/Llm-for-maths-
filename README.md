# Math-LLM-Specialist

**One-line:** Ready-to-upload GitHub template to fine-tune an open-source LLM to specialize in solving math problems (symbolic + numeric), with training, inference, evaluation, and a SymPy-backed verifier.

## Contents
- `requirements.txt`
- `Dockerfile`
- `LICENSE` (MIT)
- `data/prepare_dataset.py`
- `src/train.py`
- `src/infer.py`
- `src/math_tools.py`
- `src/eval.py`
- `.github/workflows/ci.yml`
- `.gitignore`

## Quick start (local)
```bash
git clone <your-repo-url>
cd math-llm-specialist-repo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/prepare_dataset.py --out data/processed --datasets gsm8k
# small smoke training (edit args as needed)
python src/train.py --model_name_or_path gpt2 --train_file data/processed/train.jsonl --output_dir outputs --num_train_epochs 1 --use_lora
python src/infer.py --model outputs --question "Integrate x^2 dx"
```

## License
MIT
