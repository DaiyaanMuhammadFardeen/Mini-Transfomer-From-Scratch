import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class CodeDiffDataset(Dataset):
    """
    Memory-mapped dataset. Reads token arrays directly from .npy files on disk.
    RAM usage is near-zero regardless of dataset size.

    Supports two modes:
      1. pretokenized=True  → load from tokenized_dir/*.npy (recommended)
      2. pretokenized=False → legacy in-memory mode (for small datasets only)
    """

    CHANGE_TYPE_TAGS = ['<ADD>', '<REMOVE>', '<MODIFY>',
                        '<COMMENT_ADD>', '<COMMENT_REMOVE>', '<COMMENT_MODIFY>']

    def __init__(self, tokenized_dir: str, src_vocab=None, split='train',
                 train_frac=0.9, seed=42):
        """
        Args:
            tokenized_dir: Directory containing src_tokens.npy, tgt_tokens.npy, meta.json
            src_vocab: Source vocabulary (only needed for change_features extraction)
            split: 'train' or 'val'
            train_frac: Fraction of data to use for training
            seed: Random seed for train/val split
        """
        meta_path = os.path.join(tokenized_dir, "meta.json")
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.total_rows   = self.meta["total_rows"]
        self.max_seq_length = self.meta["max_seq_length"]
        dtype_str         = self.meta.get("dtype", "int16")
        self.dtype        = np.int16 if dtype_str == "int16" else np.int32

        # Memory-mapped arrays — these DO NOT load into RAM
        self.src_mmap = np.load(
            os.path.join(tokenized_dir, "src_tokens.npy"),
            mmap_mode='r'
        )
        self.tgt_mmap = np.load(
            os.path.join(tokenized_dir, "tgt_tokens.npy"),
            mmap_mode='r'
        )

        # Deterministic train/val index split
        rng = np.random.default_rng(seed)
        all_indices = np.arange(self.total_rows)
        rng.shuffle(all_indices)
        split_pt = int(self.total_rows * train_frac)

        if split == 'train':
            self.indices = all_indices[:split_pt]
        else:
            self.indices = all_indices[split_pt:]

        # We still need src_vocab for change_features — but ONLY the stoi dict
        # Store just the integer IDs we need, not the whole vocab object
        self.change_tag_ids = []
        if src_vocab is not None:
            for tag in self.CHANGE_TYPE_TAGS:
                self.change_tag_ids.append(src_vocab.stoi.get(tag, -1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        # Read from memmap — copies only this row into RAM
        src_tokens = torch.from_numpy(
            self.src_mmap[real_idx].astype(np.int64)
        )
        tgt_tokens = torch.from_numpy(
            self.tgt_mmap[real_idx].astype(np.int64)
        )

        change_features = self._extract_change_features(src_tokens)
        return src_tokens, tgt_tokens, change_features

    def _extract_change_features(self, src_tokens):
        features = torch.zeros(6, dtype=torch.float32)
        for i, tag_id in enumerate(self.change_tag_ids):
            if tag_id != -1 and (src_tokens == tag_id).any():
                features[i] = 1.0
        return features
