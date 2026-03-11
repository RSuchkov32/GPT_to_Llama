# From GPT to Llama: A From-Scratch Implementation

Step-by-step Jupyter notebooks that convert a GPT-style transformer into Llama 2, then into Llama 3 / 3.1 / 3.2 — building every architectural component from scratch in PyTorch and loading real pretrained weights from Meta AI.

---

## Notebooks

| Notebook | Description |
|---|---|
| `converting-gpt-to-llama2.ipynb` | Converts a GPT architecture to Llama 2 by implementing RMSNorm, SiLU/SwiGLU, RoPE, and updating the attention and model classes. Loads the Llama 2 7B pretrained and chat weights. |
| `converting-llama2-to-llama3.ipynb` | Extends the Llama 2 implementation to Llama 3 by adding Grouped-Query Attention (GQA) and updated RoPE parameters. Also covers Llama 3.1 8B (with RoPE frequency scaling) and Llama 3.2 1B (with weight tying). |

---

## What You'll Learn

**Notebook 1 — GPT → Llama 2**
- Replacing LayerNorm with **RMSNorm**
- Replacing GELU with **SiLU** and implementing the **SwiGLU** feedforward variant
- Implementing **Rotary Position Embeddings (RoPE)** from scratch
- Updating `MultiHeadAttention` to use RoPE instead of absolute positional embeddings
- Loading and permuting Meta's official Llama 2 checkpoint weights into the custom architecture

**Notebook 2 — Llama 2 → Llama 3 / 3.1 / 3.2**
- How Llama 3 changes the RoPE `theta` base from 10,000 → 500,000
- Implementing **Grouped-Query Attention (GQA)** and understanding its memory savings over MHA
- The RoPE frequency scaling used in Llama 3.1 for 128k-token context
- **Weight tying** between the token embedding and output head in Llama 3.2
- Loading weights from Hugging Face `safetensors` files

---

## Requirements

**Python 3.10+** and the following packages:

```
torch>=2.1.0
sentencepiece
tiktoken
huggingface_hub
safetensors
matplotlib
nbformat
```

Install everything at once:

```bash
pip install torch sentencepiece tiktoken huggingface_hub safetensors matplotlib nbformat
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Hugging Face access token

The Llama 2 and Llama 3 weights are gated on Hugging Face. You will need to:

1. Create a [Hugging Face account](https://huggingface.co/join)
2. Request access to [meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b) and [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) (approval is usually instant)
3. Generate a [read token](https://huggingface.co/settings/tokens)
4. Create a `config.json` file in the root of the repository:

```json
{
    "HF_ACCESS_TOKEN": "hf_your_token_here"
}
```

> `config.json` is gitignored and will never be committed.

### 3. Run the notebooks

```bash
jupyter notebook
```

Run `converting-gpt-to-llama2.ipynb` first — notebook 2 imports shared components from it.

---

## Memory Requirements

| Model | float32 | bfloat16 |
|---|---|---|
| Llama 2 7B | ~26 GB | ~13 GB |
| Llama 3 8B | ~34 GB | ~17 GB |
| Llama 3.1 8B | ~34 GB | ~17 GB |
| Llama 3.2 1B | ~5 GB | ~2.5 GB |

A GPU is strongly recommended for the 7B/8B models. The Llama 3.2 1B model is small enough to run on most modern laptops in bfloat16.

---

## Architectural Changes at a Glance

```
GPT
 │
 ├─ LayerNorm          → RMSNorm
 ├─ GELU               → SiLU (SwiGLU feedforward)
 ├─ Absolute pos emb   → RoPE (applied inside attention to Q and K)
 ├─ QKV bias           → removed
 └─ Dropout            → removed
 │
Llama 2
 │
 ├─ RoPE theta base    10,000 → 500,000
 ├─ Context length     4,096  → 8,192
 ├─ Multi-Head Attn    → Grouped-Query Attention (32 Q heads, 8 KV heads)
 └─ Vocabulary         32,000 → 128,256
 │
Llama 3
 │
 ├─ Context length     8,192 → 131,072  (Llama 3.1, with RoPE freq scaling)
 └─ Weight tying + smaller model (Llama 3.2 1B)
```

---
