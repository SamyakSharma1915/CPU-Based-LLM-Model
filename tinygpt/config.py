# tinygpt/config.py
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 2000
    n_layer: int = 8
    n_head: int = 4
    d_model: int = 256
    d_ff: int = 1024
    seq_len: int = 256
    dropout: float = 0.2
    bias: bool = False
    tie_weights: bool = True

@dataclass
class TrainConfig:
    batch_size: int = 8
    grad_accum_steps: int = 1
    max_iters: int = 1500
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    val_fraction: float = 0.05
    device: str = "cpu"
    amp: bool = False

@dataclass
class SFTConfig:
    epochs: int = 2
    lr: float = 2e-4
    batch_size: int = 4
    log_every: int = 10
    pretrained_ckpt: str = "final.pt"
    output_ckpt: str = "fine_tuned.pt"
