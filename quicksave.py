# quicksave.py — Drop-in quicksave system for train.py
import os
import signal
import threading
import torch
import glob


class QuickSaver:
    """
    Saves a checkpoint every `save_every_steps` training steps.
    Also hooks SIGTERM and SIGINT to save immediately on power loss / Ctrl+C.

    Usage:
        saver = QuickSaver(
            checkpoint_dir="./checkpoints",
            save_every_steps=300,
            keep_last_n=3
        )

        # In training loop:
        saver.step(
            global_step=global_step,
            epoch=epoch,
            model=transformer,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            val_loss=val_loss,   # pass None if not yet evaluated
            best_val_loss=best_val_loss,
            use_bfloat16=use_bfloat16
        )

        # To resume:
        state = saver.load_latest()
        if state:
            transformer.load_state_dict(state['model_state_dict'])
            ...
    """

    def __init__(self, checkpoint_dir: str, save_every_steps: int = 300,
                 keep_last_n: int = 3):
        self.checkpoint_dir  = checkpoint_dir
        self.save_every_steps = save_every_steps
        self.keep_last_n     = keep_last_n
        self._lock           = threading.Lock()
        self._pending_save   = False
        self.last_saved_step = 0          # tracks when we last saved
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Hook signals — fires even on power-loss-like conditions
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT,  self._signal_handler)

        # Store references for signal handler emergency save
        self._last_state = None

    def _signal_handler(self, signum, frame):
        """Called by OS on SIGTERM / SIGINT. Saves immediately."""
        print(f"\n[QuickSaver] Signal {signum} received — saving emergency checkpoint...",
              flush=True)
        if self._last_state is not None:
            self._write_checkpoint(self._last_state, emergency=True)
        raise SystemExit(0)

    def restore_state(self, last_saved_step: int):
        """Call this after loading a checkpoint to sync internal counter."""
        self.last_saved_step = last_saved_step

    def step(self, global_step: int, epoch: int, model, optimizer,
             scheduler=None, scaler=None, val_loss=None,
             best_val_loss=float('inf'), use_bfloat16=False,
             batch_idx: int = 0):
        """Call this every training step. Saves every `save_every_steps` steps."""

        state = {
            'global_step':      global_step,
            'epoch':            epoch,
            'batch_idx':        batch_idx,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss':         val_loss,
            'best_val_loss':    best_val_loss,
            'use_bfloat16':     use_bfloat16,
        }
        if scheduler is not None:
            state['scheduler_state_dict'] = scheduler.state_dict()
        if scaler is not None:
            state['scaler_state_dict'] = scaler.state_dict()

        # Always keep last state for signal handler
        self._last_state = state

        # Use subtraction-based check (more robust than modulo after resume)
        if global_step - self.last_saved_step >= self.save_every_steps and global_step > 0:
            self._write_checkpoint(state, emergency=False)
            self.last_saved_step = global_step

    def save_epoch(self, epoch: int, model, optimizer, val_loss: float,
                   best_val_loss: float, scheduler=None, scaler=None,
                   global_step: int = 0, use_bfloat16: bool = False):
        """Call at end of each epoch for the epoch-level checkpoint."""
        state = {
            'global_step':      global_step,
            'epoch':            epoch,
            'batch_idx':        0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss':         val_loss,
            'best_val_loss':    best_val_loss,
            'use_bfloat16':     use_bfloat16,
        }
        if scheduler is not None:
            state['scheduler_state_dict'] = scheduler.state_dict()
        if scaler is not None:
            state['scaler_state_dict'] = scaler.state_dict()

        path = os.path.join(self.checkpoint_dir, f"epoch_{epoch:04d}.pth")
        self._atomic_save(state, path)
        self._last_state = state
        self._cleanup_old(prefix="step_")  # Keep epoch checkpoints, clean step ones

    def _write_checkpoint(self, state: dict, emergency: bool = False):
        with self._lock:
            step = state['global_step']
            prefix = "emergency" if emergency else f"step_{step:08d}"
            path = os.path.join(self.checkpoint_dir, f"{prefix}.pth")
            
            # Add saver's internal state to checkpoint
            state['saver_last_saved_step'] = self.last_saved_step
            
            self._atomic_save(state, path)
            if not emergency:
                self._cleanup_old(prefix="step_")

    def _atomic_save(self, state: dict, path: str):
        """Write to .tmp first, then rename. Safe against power cuts mid-write."""
        tmp_path = path + ".tmp"
        try:
            torch.save(state, tmp_path)
            os.replace(tmp_path, path)  # Atomic on Linux/Windows
            print(f"[QuickSaver] Saved: {os.path.basename(path)}", flush=True)
        except Exception as e:
            print(f"[QuickSaver] ERROR saving checkpoint: {e}", flush=True)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _cleanup_old(self, prefix: str):
        """Delete oldest step checkpoints, keeping only the last `keep_last_n`."""
        pattern = os.path.join(self.checkpoint_dir, f"{prefix}*.pth")
        files = sorted(glob.glob(pattern))  # alphabetical = chronological for step_NNNNNNNN
        while len(files) > self.keep_last_n:
            try:
                os.remove(files.pop(0))
            except OSError:
                pass

    def load_latest(self) -> dict | None:
        """
        Find the most recent checkpoint in checkpoint_dir.
        Returns the loaded state dict, or None if no checkpoint exists.
        Prefers step checkpoints over epoch checkpoints for finest granularity.
        """
        # Try step checkpoints first (finest granularity)
        step_files = sorted(glob.glob(
            os.path.join(self.checkpoint_dir, "step_*.pth")
        ))
        epoch_files = sorted(glob.glob(
            os.path.join(self.checkpoint_dir, "epoch_*.pth")
        ))
        emergency_files = glob.glob(
            os.path.join(self.checkpoint_dir, "emergency.pth")
        )

        candidates = step_files + epoch_files + emergency_files
        if not candidates:
            return None

        # Pick the one with the highest global_step
        best_path  = None
        best_step  = -1
        for path in candidates:
            try:
                ckpt = torch.load(path, map_location='cpu')
                gs = ckpt.get('global_step', 0)
                if gs > best_step:
                    best_step = gs
                    best_path = path
            except Exception:
                continue  # Skip corrupted files

        if best_path is None:
            return None

        print(f"[QuickSaver] Resuming from: {best_path} (global_step={best_step})")
        return torch.load(best_path, map_location='cpu')
