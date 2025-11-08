# Transformer-based CodeDiff → Commit Message Model

## 🧠 Overview

This model is a **Transformer sequence-to-sequence** model trained to generate commit messages (or text summaries) from code diffs.
It uses the **multi-headed self-attention mechanism** introduced in *"Attention Is All You Need"* (Vaswani et al., 2017).
The model learns relationships between code changes (diffs) and their corresponding messages using encoder–decoder attention.

---

## ⚙️ Training Configuration

| Hyperparameter   | Value                 | Description                                         |
| ---------------- | --------------------- | --------------------------------------------------- |
| `d_model`        | **1024**              | Dimension of the embeddings and model hidden size   |
| `num_heads`      | **8**                 | Number of attention heads in multi-headed attention |
| `num_layers`     | **4**                 | Number of encoder and decoder transformer layers    |
| `d_ff`           | **2048**              | Feed-forward hidden dimension                       |
| `max_seq_length` | **256**               | Maximum sequence length of tokenized inputs         |
| `dropout`        | **0.3**               | Dropout probability                                 |
| `batch_size`     | **32**                | Number of samples per batch                         |
| `num_epochs`     | **5**                 | Number of full training epochs                      |
| `learning_rate`  | **1e-5**              | Learning rate for Adam optimizer                    |
| `optimizer`      | **Adam**              | Adaptive optimizer used for training                |
| `scheduler`      | **ReduceLROnPlateau** | Learning rate scheduler                             |
| `criterion`      | **CrossEntropyLoss**  | Loss function used for training                     |

---

## 📊 Dataset Details

* **Training samples:** 58,874
* **Validation samples:** 6,542
* **Total:** 65,416 pairs of diffs and commit messages.
* Source vocabulary size: **34,576**
* Target vocabulary size: **5,000**

Data was loaded from:

```
./train_data.parquet
```

and tokenized using:

```
./tokenizer/diff_vocab.pkl
./tokenizer/message_vocab.pkl
```

---

## 🛠️ Model Architecture

The model follows a **Transformer Encoder–Decoder** architecture:

* **Encoder:** processes the diff tokens into contextual embeddings.
* **Decoder:** attends to encoder outputs to generate message tokens sequentially.
* Each block includes:

  * Multi-headed self-attention layer
  * Feed-forward sub-layer
  * Residual connections and layer normalization
* The model contains approximately **129,631,112 parameters**.

---

## 🚀 Training Environment

* **Device:** CUDA (GPU)
* **Precision:** Automatic Mixed Precision (GradScaler used)
* **Framework:** PyTorch
* **Checkpointing:** Model saved to `model.pth` every epoch.
* **Training script:** `train.py`

---

## 💿 Checkpoints

If training was resumed, checkpoints were expected in:

```
./checkpoints/transformer_epoch_X.pth
```

The final trained model is saved as:

```
model.pth
```

---

## 🤩 Inference Instructions

### 1. Requirements

```bash
pip install torch
```

### 2. Load the model

```python
import torch
from model_definition import TransformerModel  # define or import your model class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = TransformerModel(
    d_model=1024,
    num_heads=8,
    num_layers=4,
    d_ff=2048,
    max_seq_length=256,
    dropout=0.3,
    src_vocab_size=34576,
    tgt_vocab_size=5000
).to(device)

model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
```

### 3. Run inference

```python
# Tokenize input diff (example)
input_ids = diff_tokenizer.encode("<your code diff>")
input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)

# Generate prediction (greedy example)
with torch.no_grad():
    output = model.generate(input_tensor, max_length=50)

message = msg_tokenizer.decode(output[0].tolist())
print("Generated message:", message)
```

---

## 🧠 Notes

* The model uses **multi-headed attention** for capturing complex relationships between code tokens and message tokens.
* **Dropout** (0.3) was used to reduce overfitting.
* **Sequence length** was capped at 256, as the average diff length is 128 and average message length is 12.

---

## 📅 Author / Project Info

* **Author:** Daiyaan Muhammad Fardeen
* **Framework:** PyTorch
* **Last Trained:** November 2025
* **Use Case:** CodeDiff-to-Message generation
* **Model file:** `model.pth`
* **Tokenizer:** `tokenizer/diff_vocab.pkl` and `tokenizer/message_vocab.pkl`

---

> 💡 *This model directory is self-contained and intended for inference or further fine-tuning. Keep the `.pth` file and tokenizer files together to ensure consistent vocab indexing during inference.*

