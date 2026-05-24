# tinygpt/finetune.py
import os
import time
import torch
from torch.utils.data import DataLoader
from .config import SFTConfig, ModelConfig
from .dataset import SFTDataset
from .tokenizer import Tokenizer
from .model import GPT


def resolve_pretrained_checkpoint(path):
    if os.path.exists(path):
        return path
    fallback = "final.pt"
    if path == "best.pt" and os.path.exists(fallback):
        print(f"[sft] pretrained checkpoint {path} not found, using {fallback} instead")
        return fallback
    raise FileNotFoundError(
        f"Pretrained checkpoint not found: {path}. "
        "Train the model first or update SFTConfig.pretrained_ckpt."
    )

def finetune(sft_cfg: SFTConfig, model_cfg: ModelConfig = ModelConfig()):
    """
    SFT fine-tuning tool.

    JSONL format:
    {"prompt": "...", "response": "..."}
    Each line is one example.

    Prompt masking semantics:
    - `encode_chat` sets prompt tokens in target sequence to -1.
    - CE loss uses ignore_index=-1 so only response tokens are optimized.
    """
    device = torch.device("cpu")
    tokenizer = Tokenizer()
    dataset = SFTDataset(tokenizer, chat_path="data/chat_finetune.jsonl", seq_len=model_cfg.seq_len)
    loader = DataLoader(dataset, batch_size=sft_cfg.batch_size, shuffle=True)
    total_batches = len(loader)

    model = GPT(model_cfg).to(device)
    ckpt_path = resolve_pretrained_checkpoint(sft_cfg.pretrained_ckpt)
    print(f"[sft] loading pretrained checkpoint from {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    print(
        f"[sft] dataset_examples={len(dataset)} batch_size={sft_cfg.batch_size} "
        f"batches_per_epoch={total_batches} epochs={sft_cfg.epochs}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=sft_cfg.lr)
    for epoch in range(sft_cfg.epochs):
        model.train()
        total = 0.0
        count = 0
        window_total = 0.0
        window_count = 0
        epoch_start = time.time()
        for step, (x, y) in enumerate(loader, start=1):
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item() * x.size(0)
            count += x.size(0)
            window_total += loss.item() * x.size(0)
            window_count += x.size(0)

            if step % sft_cfg.log_every == 0 or step == total_batches:
                elapsed = time.time() - epoch_start
                running_avg = window_total / max(1, window_count)
                epoch_avg = total / max(1, count)
                pct = (step / max(1, total_batches)) * 100
                print(
                    f"[sft] epoch={epoch+1}/{sft_cfg.epochs} "
                    f"step={step}/{total_batches} pct={pct:.1f}% "
                    f"running_loss={running_avg:.4f} avg_loss={epoch_avg:.4f} "
                    f"elapsed={elapsed:.1f}s"
                )
                window_total = 0.0
                window_count = 0
        avg = total / max(1, count)
        print(f"[sft] epoch={epoch+1}/{sft_cfg.epochs} complete avg_loss={avg:.4f}")

    os.makedirs(os.path.dirname(sft_cfg.output_ckpt) or ".", exist_ok=True)
    torch.save(model.state_dict(), sft_cfg.output_ckpt)
    print(f"[sft] saved fine-tuned checkpoint to {sft_cfg.output_ckpt}")

if __name__ == "__main__":
    config = SFTConfig()
    finetune(config)
