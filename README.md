# Mini-Transfomer-From-Scratch
Trying to create an efficient mini transformer from scratch by introducing efficiency techniques from recent 2024 and 2025 papers. I mostly don't know what I am doing most of the time. But I will attempt anyway

# 🧠 Transformer Implementation Progress Log
---

### 📅 **Day 1 — Initial Setup & Modularization**
**Date:** Sunday, October 5, 2025

#### ✅ What I Did
- Followed this [Medium article](https://medium.com/data-science/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb) to implement a basic transformer model from scratch using pytorch's nn module.
- Copied the full code implementation and **modularized** each component into separate files for better configurability and clarity.  
- Training runs on the GPU memory

#### 🧩 Issues Faced
- Tried to modularize the code but faced import issues 
- Mask generation was being done on the CPU with RAM

#### 🛠️ How I Fixed It
- Fixed the import issues by making import corrections
- Mask matricies are now created in GPU memory in parallel

#### 💡 What I Learned
- Learned how to implement a basic transformer model from scratch using pytorch's nn module.

#### 🎯 Next Steps
- Improve the train.py code to train using GPU and add progress bars and loggers.
---

📅 Day 2 — Dataset Integration, Custom Tokenizer, and Training Enhancements
Date: Wednesday, October 8, 2025
#### ✅ What I Did

- Integrated my own dataset (dataset_part2.parquet) into the training pipeline, replacing the default dataset.
- Implemented a CodeDiffDataset and Vocabulary class in customTokenizer.py to handle tokenization of code diff data, including special tokens for diffs (e.g., <ADD>, <REMOVE>).
- Created an evaluation script for inference to test the trained Transformer model.
- Optimized hyperparameters (src/tgt_vocab_size=5000, d_model=256, num_layers=6, num_heads=4, d_ff=1024, max_seq_length=256, batch_size=1) to fully utilize the 8 GB VRAM of my RX 6600 GPU without OOM errors.
- Configured customTokenizer.py to export tokenized vocabularies to disk (src_vocab.pkl, tgt_vocab.pkl) for faster training by skipping repeated tokenization.
- Added tqdm progress bars to both train.py (for epochs and batches) and customTokenizer.py (for vocabulary building) to track progress and ETA.
- Parallelized tokenization in customTokenizer.py to use all available CPU cores via multiprocessing for faster vocabulary building.

#### 🧩 Issues Faced

- Encountered an IndexError: index out of range in self in the embedding layer due to vocabulary indices exceeding src/tgt_vocab_size=5000.
- Faced a ROCm HSA_STATUS_ERROR_EXCEPTION: code 0x1016 on the GPU due to excessive VRAM usage with initial hyperparameters (max_seq_length=4096, batch_size=16).
- Lacked visibility into training progress, making it hard to estimate completion time.

#### 🛠️ How I Fixed It

- Fixed the IndexError by sorting vocabulary by frequency in customTokenizer.py and truncating src_vocab and tgt_vocab to 5000 tokens in train.py, ensuring indices stay within bounds.
- Resolved the ROCm OOM error by reducing max_seq_length to 256 and batch_size to 1, fitting training within ~3 GB VRAM (verified with theoretical estimates and torch.cuda.memory_allocated()).
- Added tqdm progress bars in train.py for epoch and batch progress, displaying batch loss and ETA, and in customTokenizer.py for tokenization and vocabulary building.

#### 💡 What I Learned

- Learned how to build a custom tokenizer for code diff data, handling special tokens and parallelizing across CPU cores for efficiency.
- Gained experience optimizing Transformer hyperparameters to fit within GPU VRAM constraints (8 GB on RX 6600).
- Understood the importance of progress tracking with tqdm for long-running training and tokenization tasks.
- Learned to create an evaluation script for Transformer inference and manage vocabulary persistence to disk for faster workflows.

#### 🎯 Next Steps

- Implement memory optimizations in train.py (e.g., torch.cuda.amp for mixed-precision, gradient checkpointing) to allow larger batch_size or max_seq_length.
- Enhance the evaluation script with metrics (e.g., BLEU, accuracy) and integrate it into the training loop for periodic validation.
- Profile and optimize tokenization speed further, possibly caching tokenized data for even faster loading.

---
