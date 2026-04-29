import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class SwiGLU(nn.Module):
    """Corrected SwiGLU — LLaMA / Mistral style with proper SiLU activation."""
    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        # Note: true SwiGLU uses d_ff * 2/3 to keep param count equivalent to vanilla FFN
        # Round to nearest multiple of 64 for CUDA efficiency
        d_ff_swiglu = int(d_ff * 2 / 3)
        d_ff_swiglu = (d_ff_swiglu + 63) // 64 * 64   # Round up to multiple of 64
        self.gate = nn.Linear(d_model, d_ff_swiglu, bias=bias)
        self.up   = nn.Linear(d_model, d_ff_swiglu, bias=bias)
        self.down = nn.Linear(d_ff_swiglu, d_model, bias=bias)

    def forward(self, x):
        # SiLU gate × value projection — the true SwiGLU formula
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))

class GeGLU(nn.Module):
    """PaLM-2 / DeepSeek-Coder"""
    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_model, d_ff, bias=bias)
        self.w3 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.w3(self.w2(x) * F.gelu(self.w1(x), approximate='tanh'))

class ReGLU(nn.Module):
    """Fastest on CPU"""
    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_model, d_ff, bias=bias)
        self.w3 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.w3(self.w2(x) * F.relu(self.w1(x)))
