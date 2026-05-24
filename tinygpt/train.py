# tinygpt/train.py
import math
import os
import torch
from torch.utils.data import DataLoader
from .config import TrainConfig, ModelConfig
from .dataset import PretrainDataset
from .tokenizer import Tokenizer
from .model import GPT


torch.set_num_threads(4)          
torch.set_float32_matmul_precision('high')

def get_lr_multiplier(step, cfg: TrainConfig):
    if step < cfg.warmup_iters:
        return step / max(1, cfg.warmup_iters)
    progress = (step - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

def train():
    cfg = TrainConfig()
    model_cfg = ModelConfig()
    device = torch.device(cfg.device)
    tokenizer = Tokenizer()
    train_dataset = PretrainDataset(
        tokenizer,
        corpus_path="data/corpus.txt",
        val_fraction=cfg.val_fraction,
        seq_len=model_cfg.seq_len,
        split="train",
    )
    val_dataset = PretrainDataset(
        tokenizer,
        corpus_path="data/corpus.txt",
        val_fraction=cfg.val_fraction,
        seq_len=model_cfg.seq_len,
        split="val",
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    model = GPT(model_cfg).to(device)

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "bias" not in name.lower() and "ln" not in name.lower():
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
    )

    scaler = torch.cuda.amp.GradScaler() if cfg.amp else None
    autocast = torch.cuda.amp.autocast if cfg.device != "cpu" else torch.cpu.amp.autocast

    best_val_loss = float("inf")
    global_step = 0

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1000):
        model.train()
        running_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            global_step += 1
            x = x.to(device)
            y = y.to(device)

            with autocast(enabled=cfg.amp):
                logits, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps

            if cfg.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if global_step % cfg.grad_accum_steps == 0:
                lr_mult = get_lr_multiplier(global_step, cfg)
                lr = cfg.min_lr + (cfg.lr - cfg.min_lr) * lr_mult
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                if cfg.amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                if cfg.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()

            running_loss += loss.item() * cfg.grad_accum_steps

            if global_step % 100 == 0:
                avg = running_loss / 100
                print(f"[train] step={global_step} epoch={epoch} loss={avg:.4f} lr={lr:.5g}")
                running_loss = 0.0

            if global_step % 200 == 0:
                model.eval()
                val_loss = 0.0
                val_count = 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx = vx.to(device)
                        vy = vy.to(device)
                        with autocast(enabled=cfg.amp):
                            _, vloss = model(vx, vy)
                        val_loss += vloss.item() * vx.size(0)
                        val_count += vx.size(0)
                val_loss = val_loss / max(val_count, 1)
                print(f"[val] step={global_step} val_loss={val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), "best.pt")
                    print(f"[val] saved best.pt (val_loss={val_loss:.4f})")
                model.train()

            if global_step % 1000 == 0:
                ckpt_path = f"checkpoints/checkpoint_{global_step}.pt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"[save] step={global_step} checkpoint saved to {ckpt_path}")

            if global_step >= cfg.max_iters:
                break
        if global_step >= cfg.max_iters:
            break

    torch.save(model.state_dict(), "final.pt")
    print("Training complete.")

if __name__ == "__main__":
    train()
