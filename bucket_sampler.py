# bucket_sampler.py
import numpy as np
from torch.utils.data import Sampler


class BucketBatchSampler(Sampler):
    """
    Groups sequences by length into buckets, then batches within each bucket.
    Reduces padding by 40-70% compared to random batching.

    Args:
        lengths:      Array/list of sequence lengths (one per sample)
        batch_size:   Number of samples per batch
        bucket_size:  How many samples to sort at a time (larger = better bucketing,
                      but less shuffling within epoch). Rule of thumb: 100 × batch_size
        drop_last:    Drop incomplete final batch
    """

    def __init__(self, lengths, batch_size: int, bucket_size: int = None,
                 drop_last: bool = False, shuffle: bool = True):
        self.lengths    = np.array(lengths)
        self.batch_size = batch_size
        self.bucket_size = bucket_size or min(len(lengths), batch_size * 100)
        self.drop_last  = drop_last
        self.shuffle    = shuffle

    def __iter__(self):
        indices = np.random.permutation(len(self.lengths)) if self.shuffle \
                  else np.arange(len(self.lengths))

        batches = []
        for bucket_start in range(0, len(indices), self.bucket_size):
            bucket_idx = indices[bucket_start : bucket_start + self.bucket_size]
            # Sort this bucket by length
            bucket_sorted = bucket_idx[np.argsort(self.lengths[bucket_idx])]
            # Split into batches
            for i in range(0, len(bucket_sorted), self.batch_size):
                batch = bucket_sorted[i : i + self.batch_size].tolist()
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

        # Shuffle the order of batches (so model doesn't always see short seqs first)
        if self.shuffle:
            np.random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


class CurriculumSampler(BucketBatchSampler):
    """
    First `warmup_epochs` epochs: sample only from the easiest (shortest) 50% of data.
    After that: sample from all data with bucket batching.
    """

    def __init__(self, lengths, batch_size, warmup_epochs=1,
                 current_epoch=0, **kwargs):
        self.warmup_epochs   = warmup_epochs
        self.current_epoch   = current_epoch
        self.all_lengths     = np.array(lengths)
        super().__init__(lengths, batch_size, **kwargs)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
        if epoch < self.warmup_epochs:
            # Use only shortest 50% of sequences
            threshold = np.percentile(self.all_lengths, 50)
            mask = self.all_lengths <= threshold
            self.lengths = self.all_lengths[mask]
            print(f"[Curriculum] Epoch {epoch}: training on {mask.sum()} short sequences "
                  f"(≤{threshold:.0f} tokens)")
        else:
            self.lengths = self.all_lengths
            print(f"[Curriculum] Epoch {epoch}: full dataset ({len(self.lengths)} samples)")
