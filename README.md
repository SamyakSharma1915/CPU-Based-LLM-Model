<div align="center">

### Lightweight CPU-Based LLM Framework Built with PyTorch

A minimal transformer-based Large Language Model designed for  
experimentation, tokenizer training, fine-tuning, and local inference  
on low-resource systems.

</div>

---

# ✨ Features

- ⚡ Lightweight transformer-based architecture
- 🧠 CPU-friendly training and inference
- 🔤 Custom tokenizer using SentencePiece
- 📚 Fine-tuning support with JSONL datasets
- 💬 Interactive terminal chat interface
- 🛠 Fully configurable training pipeline
- 🎯 Educational and research-focused implementation

---

# 🧰 Tech Stack

| Technology | Usage |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning framework |
| SentencePiece | Tokenizer training |

---

# 📦 Installation

```bash
pip install torch sentencepiece
```

---

# 📚 Dataset Preparation

Merge all markdown/text files into a single training corpus:

```bash
cat *.md > data/corpus.txt
```

---

# 🚀 Training Pipeline

## 1️⃣ Train Tokenizer

```bash
python -m tinygpt.tokenizer
```

---

## 2️⃣ Pretrain Model

```bash
python -m tinygpt.train
```

---

## 3️⃣ Fine-tune Model

```bash
python -m tinygpt.finetune
```

---

## 4️⃣ Start Chat Interface

```bash
python -m tinygpt.chat
```

---

# 🧪 Quick Sanity Check

Edit `tinygpt/config.py`

```python
max_iters = 200
```

### Expected Result

- Training loss should fall below `4.0`
- Usually within ~50 training steps

---

# ⚙️ Training Notes

If validation loss diverges:

```python
dropout = 0.2
d_model = 256
tie_weights = True
```

### Recommended Fixes

- Increase dropout
- Reduce model dimension
- Add more training data
- Keep tied embeddings enabled

---

# 🗂 Dataset Information

| File | Description |
|---|---|
| `data/corpus.txt` | Plain English source text |
| `data/chat_finetune.jsonl` | Prompt/response fine-tuning dataset |
| `newfinetune.jsonl` | Large conversational dataset |
| `0000.json` | Internet-sourced prompt/response dataset |
| `data/fullEnglish` | Wikipedia-based English dataset |
| `build_large_corpus` | Synthetic definition generator |
| `tokenizer/spm.model` | Generated tokenizer model |

---

# 📜 License

Licensed under the **Apache License 2.0**

See the `LICENSE` file for more information.

---

# © Copyright

```text
© 2026 Samyak Sharma. All Rights Reserved.
```

### Attribution Notice

The original implementation, architecture, documentation, and custom modifications were created by **Samyak Sharma**.

Proper attribution is appreciated when using or modifying this project.

---

# ⚠ Disclaimer

This project is intended for:

- Educational purposes
- AI research
- Experimental transformer architectures
- Lightweight local LLM development

Some datasets may contain publicly available internet text and generated conversational data.

---

<div align="center">

# Author

## Samyak Sharma

</div>
>>>>>>> 136f57f437e004e759eec3857de7f26049c53de7
