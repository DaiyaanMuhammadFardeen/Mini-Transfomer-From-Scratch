# ema.py
import copy
import torch


class ModelEMA:
    """
    Maintains a moving average of model parameters.
    Use ema.model for inference/validation — it typically outperforms the raw model.

    decay: typically 0.999 – 0.9999. Higher = slower update = smoother weights.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        # Create a deep copy on CPU to save GPU memory
        self.model = copy.deepcopy(model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        """Call after every optimizer step."""
        for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
