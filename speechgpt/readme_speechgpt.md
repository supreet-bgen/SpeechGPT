# SpeechGPT Setup & Usage

## 1. Poetry Setup
Install [pipx](https://pipx.pypa.io/stable/) and use it to install Poetry:
```bash
pip install pipx
pipx install poetry
```

## 2. Install Dependencies
From the project root:

```bash
poetry install
```

## 3. Folder Structure
For the inference and evaluation scripts to work correctly, your project directory should be structured as follows.

```
SpeechGPT/
├── cli_infer.py
├── utils/
│   ├── speech2unit/
│   └── vocoder/
├── SpeechGPT-7B-cm/
│   └── (model files here)
├── SpeechGPT-7B-com/
│   └── (LoRA weights here)
└── VoxEval/
    └── VoxEval_evaluation.py
```

## 4. Inference using CLI
Run inference with SpeechGPT:

### set required paths

```bash
s2u_dir=utils/speech2unit/
vocoder_dir=utils/vocoder/
```
### run inference
```bash

python cli_infer.py \
    --model-name-or-path "SpeechGPT-7B-cm" \
    --lora-weights "SpeechGPT-7B-com" \
    --s2u-dir "$s2u_dir" \
    --vocoder-dir "$vocoder_dir" \
    --output-dir "output"
```

5. Evaluation using VoxEval
Run evaluation with VoxEval:

```bash

poetry run python ../../VoxEval/VoxEval_evaluation.py \
    --eval_slm_path /Work/SpeechGPT/speechgpt/ \
    --e2e_eval_file e2e_eval \
    --save_folder output_eval2
```

# Discussion

### Tokenizer Discussion

**Hypothesis:** The initial hypothesis was that during the inference stage, the *HuBERT speech tokens* were being processed as standard text by the *Llama* tokenizer. This could lead to a fundamental mismatch between the input type and the model's expected token format.

**Verification:** A review of the training code verified that this was not the case. The tokenizer's vocabulary is explicitly expanded during training to properly handle the speech tokens.

The following code snippet confirms this process:

```Python

if '<sosp>' not in tokenizer.get_vocab():
    units_size=1000
    logger.info(f"Add special unit tokens <0>-<{units_size-1} to tokenizer.vocab")
    new_tokens = [f"<{x}>" for x in range(units_size)] + ['<sosp>', '<eosp>']
    tokenizer.add_tokens(new_tokens)
```
This code adds 1000 unique tokens (e.g., <0>, <1>) that represent the discrete HuBERT speech units, along with special start-of-speech (<sosp>) and end-of-speech (<eosp>) tokens. This process ensures the model correctly interprets the speech input, invalidating the initial hypothesis.