import torch
import torch.nn as nn

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, head_dim: int = None, base: int = 10000):
        """
        head_dim: optional expected per-head dimension. If provided, a cached inv_freq is registered.
                  If None, inv_freq will be computed dynamically from the input tensor shape.
        base: rotary base
        """
        super().__init__()
        if head_dim is not None:
            assert head_dim % 2 == 0, f"head_dim={head_dim} must be even"
        self.head_dim = head_dim
        self.base = base

        if head_dim is not None:
            # Frequencies for head_dim (only if caller gave a head_dim)
            inv_freq = 1.0 / (
                base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            )
            # Keep as buffer for speed if head_dim was supplied
            self.register_buffer('inv_freq', inv_freq, persistent=False)
        else:
            # sentinel; when None, we will compute inv_freq on demand
            self.register_buffer('inv_freq', torch.tensor([], dtype=torch.float32), persistent=False)

    def _make_inv_freq_for_half(self, half: int, device):
        """Create inv_freq for a half-dimension of size `half` (i.e. head_dim = half*2)."""
        # head_dim = half * 2
        head_dim = half * 2
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
        )
        return inv_freq  # length = half

    def apply_rotary_emb(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        x: [batch, heads, seq_len, head_dim] or [*, seq_len, head_dim]
        Returns rotated tensor same shape as x.
        """
        half = x.shape[-1] // 2
        if half * 2 != x.shape[-1]:
            raise ValueError(f"input last-dim must be even, got {x.shape[-1]}")

        seq_len = min(seq_len, x.shape[-2])  # 🔒 ensure alignment

        if self.head_dim is not None and getattr(self, "inv_freq", None) is not None and self.inv_freq.numel() == half:
            inv_freq = self.inv_freq.to(x.device)
        else:
            inv_freq = self._make_inv_freq_for_half(half, device=x.device)

        pos = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        angles = pos.unsqueeze(1) * inv_freq.unsqueeze(0)

        sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
        cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)

        x1 = x[..., :half]
        x2 = x[..., half:]

        # Match shapes safely
        min_half = min(x1.shape[-1], cos.shape[-1])
        x1, x2 = x1[..., :min_half], x2[..., :min_half]
        cos, sin = cos[..., :min_half], sin[..., :min_half]

        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated

    # kept for compatibility
    def _rotate_half(self, x):
        """Rotate half the hidden dimensions"""
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, x, seq_len):
        return self.apply_rotary_emb(x, seq_len)

