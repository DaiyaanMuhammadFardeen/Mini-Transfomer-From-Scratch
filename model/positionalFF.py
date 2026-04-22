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
    """LLaMA-2 / CodeLlama / Mistral default"""
    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)  # gate
        self.w2 = nn.Linear(d_model, d_ff, bias=bias)  # value
        self.w3 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.w3(self.w2(x) * torch.sigmoid(self.w1(x)))

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
