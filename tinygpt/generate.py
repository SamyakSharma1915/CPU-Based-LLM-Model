# tinygpt/generate.py
import json
import os
import re
import torch
from .config import ModelConfig
from .model import GPT
from .tokenizer import Tokenizer

def _top_k_top_p_filtering(logits, top_k=0, top_p=1.0):
    top_k = min(max(top_k, 0), logits.size(-1))
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_values, torch.full_like(logits, -1e10), logits)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 0] = False
        remove_mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(remove_mask, -1e10)
    return logits


def _apply_repetition_penalty(logits, generated_ids, penalty=1.15, recent_window=64):
    if penalty <= 1.0 or not generated_ids:
        return logits
    unique_ids = set(generated_ids[-recent_window:])
    for token_id in unique_ids:
        logits[:, token_id] /= penalty
    return logits


def _count_sentence_endings(text):
    return len(re.findall(r"[.!?]+", text))


def clean_response_text(text, prompt=""):
    cleaned = text.replace("\r", "")

    if prompt and cleaned.startswith(prompt):
        cleaned = cleaned[len(prompt):]

    if "<|assistant|>" in cleaned:
        cleaned = cleaned.split("<|assistant|>")[-1]
    if "<|user|>" in cleaned:
        cleaned = cleaned.split("<|user|>")[0]

    cleaned = cleaned.replace("<|assistant|>", "").replace("<|user|>", "").replace("<|system|>", "")
    cleaned = cleaned.replace("Assistant:", "").strip()

    if prompt:
        prompt_plain = prompt.replace("<|user|>", "").replace("<|assistant|>", "").strip()
        if prompt_plain and cleaned.startswith(prompt_plain):
            cleaned = cleaned[len(prompt_plain):].strip(" :\n\t")

    cleaned = re.sub(r"\b(\w+)(\s+\1\b){2,}", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def sample(model, tok, prompt, max_new_tokens=64, temperature=1.0, top_k=50, top_p=0.95, json_mode=False):
    model.eval()
    device = next(model.parameters()).device
    raw_prompt = prompt
    valid_vocab_size = getattr(tok, "vocab_size", None)
    min_new_tokens = min(48, max_new_tokens)
    min_sentences = 2
    if json_mode:
        instruction = {"instruction": "Respond with a valid JSON object.", "input": prompt}
        prompt = json.dumps(instruction, ensure_ascii=False)

    input_ids = tok.encode(prompt, add_bos=True, add_eos=False)
    generated_ids = list(input_ids)

    brace_depth = 0
    prefix_text = tok.decode(input_ids)
    for ch in prefix_text:
        if ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1

    generated_token_count = 0
    for _ in range(max_new_tokens):
        idx_cond = torch.tensor([generated_ids[-model.cfg.seq_len :]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(1e-9, temperature)
        if valid_vocab_size is not None and valid_vocab_size < logits.size(-1):
            logits = logits[:, :valid_vocab_size]
        logits = _apply_repetition_penalty(logits, generated_ids)

        partial_text = clean_response_text(tok.decode(generated_ids), raw_prompt)
        if (
            not json_mode
            and generated_token_count < max_new_tokens
            and (
                generated_token_count < min_new_tokens
                or _count_sentence_endings(partial_text) < min_sentences
            )
            and tok.eos_id < logits.size(-1)
        ):
            logits[:, tok.eos_id] = -1e10

        logits = _top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        if generated_ids and next_id == generated_ids[-1]:
            logits[:, next_id] = -1e10
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

        generated_ids.append(next_id)
        generated_token_count += 1

        if next_id == tok.eos_id:
            break

        token_text = tok.decode([next_id])
        for ch in token_text:
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
        if json_mode and brace_depth <= 0 and "}" in token_text:
            break

    result_text = tok.decode(generated_ids)
    if json_mode:
        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            return result_text
    return clean_response_text(result_text, raw_prompt)


def resolve_checkpoint_path(ckpt_path):
    if os.path.exists(ckpt_path):
        return ckpt_path

    fallbacks = []
    if ckpt_path == "best.pt":
        fallbacks = ["fine_tuned.pt", "final.pt"]
    elif ckpt_path == "fine_tuned.pt":
        fallbacks = ["final.pt"]

    for fallback in fallbacks:
        if os.path.exists(fallback):
            print(f"[load] checkpoint {ckpt_path} not found, using {fallback} instead")
            return fallback

    raise FileNotFoundError(
        f"Checkpoint not found: {ckpt_path}. "
        "Available defaults are fine_tuned.pt, final.pt, or a custom --ckpt path."
    )


def load_model(ckpt_path, device="cpu"):
    cfg = ModelConfig()
    model = GPT(cfg).to(device)
    resolved_path = resolve_checkpoint_path(ckpt_path)
    model.load_state_dict(torch.load(resolved_path, map_location=device))
    model.eval()
    tokenizer = Tokenizer("tokenizer/spm.model")
    return model, tokenizer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="fine_tuned.pt")
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max", type=int, default=256)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    m, t = load_model(args.ckpt, device="cpu")
    print(sample(m, t, args.prompt, max_new_tokens=args.max, temperature=args.temp, top_k=args.top_k, top_p=args.top_p, json_mode=args.json))
