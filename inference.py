import torch
import pickle
import argparse
import os
from model.model import Transformer
from tokenizer.DiffVocabulary import DiffVocabulary
from tokenizer.MsgVocabulary import MsgVocabulary

def load_pickle_with_redirect(file_path):
    """Load pickle file with module path redirection."""
    class PickleModuleRedirector(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'Vocabulary':
                try:
                    from tokenizer.DiffVocabulary import DiffVocabulary
                    if name == 'DiffVocabulary' or name == 'Vocabulary':
                        return DiffVocabulary
                except:
                    pass
                try:
                    from tokenizer.MsgVocabulary import MsgVocabulary
                    if name == 'MsgVocabulary' or name == 'Vocabulary':
                        return MsgVocabulary
                except:
                    pass
            elif module in ('DiffVocabulary', 'tokenizer.diff_text.Vocabulary'):
                from tokenizer.DiffVocabulary import DiffVocabulary
                return DiffVocabulary
            elif module in ('MsgVocabulary', 'tokenizer.message.Vocabulary'):
                from tokenizer.MsgVocabulary import MsgVocabulary
                return MsgVocabulary
            return super().find_class(module, name)
    
    with open(file_path, 'rb') as f:
        unpickler = PickleModuleRedirector(f)
        try:
            obj = unpickler.load()
            return obj
        except Exception as e:
            print(f"Failed to load with redirection, trying standard pickle: {e}")
            f.seek(0)
            obj = pickle.load(f)
            return obj

def generate_message(model, src_tokens, tgt_vocab, device, max_length=50):
    """Generate a commit message using the trained model."""
    model.eval()
    src_tokens = src_tokens.unsqueeze(0).to(device)  # Add batch dimension
    
    # Start with SOS token
    tgt_tokens = torch.tensor([[tgt_vocab.stoi["<SOS>"]]], dtype=torch.long).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Generate masks
            src_mask, tgt_mask = model.generate_mask(src_tokens, tgt_tokens)
            
            # Forward pass
            output = model(src_tokens, tgt_tokens)
            
            # Get the prediction for the last token
            pred = output[:, -1, :]
            next_token = pred.argmax(dim=-1).unsqueeze(0)
            
            # Append prediction to target sequence
            tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)
            
            # Stop if EOS token is generated
            if next_token.item() == tgt_vocab.stoi["<EOS>"]:
                break
    
    # Convert tokens to text
    message_tokens = tgt_tokens.squeeze(0).cpu().numpy()
    message = []
    for token_id in message_tokens:
        if token_id == tgt_vocab.stoi["<EOS>"]:
            break
        if token_id not in [tgt_vocab.stoi["<PAD>"], tgt_vocab.stoi["<SOS>"]]:
            message.append(tgt_vocab.itos[token_id])
    
    return " ".join(message)

def main():
    parser = argparse.ArgumentParser(description="Inference script for Transformer model")
    parser.add_argument("--model-path", default="transformer_model.pth", help="Path to trained model")
    parser.add_argument("--diff-vocab-path", default="./tokenizer/diff_vocab.pkl", help="Path to diff vocabulary")
    parser.add_argument("--message-vocab-path", default="./tokenizer/message_vocab.pkl", help="Path to message vocabulary")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--d-ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--max-seq-length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--diff-text", type=str, required=True, help="Input diff text for inference")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabularies
    print("Loading vocabularies...")
    if not os.path.exists(args.diff_vocab_path):
        args.diff_vocab_path = "./diff_vocab.pkl"
    if not os.path.exists(args.message_vocab_path):
        args.message_vocab_path = "./message_vocab.pkl"
        
    src_vocab = load_pickle_with_redirect(args.diff_vocab_path)
    tgt_vocab = load_pickle_with_redirect(args.message_vocab_path)
    
    src_vocab_size = len(src_vocab.stoi)
    tgt_vocab_size = len(tgt_vocab.stoi)
    print(f"Source vocab size: {src_vocab_size}")
    print(f"Target vocab size: {tgt_vocab_size}")
    
    # Initialize model
    print("Initializing model...")
    model = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        args.d_model,
        args.num_heads,
        args.num_layers,
        args.d_ff,
        args.max_seq_length,
        args.dropout
    ).to(device)
    
    # Load trained model weights
    print(f"Loading model from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("Model loaded successfully!")
    
    # Tokenize input diff text
    print("Tokenizing input...")
    diff_tokens = src_vocab.numericalize(args.diff_text)
    diff_tokens = [src_vocab.stoi["<SOS>"]] + diff_tokens + [src_vocab.stoi["<EOS>"]]
    
    # Pad or truncate to max_seq_length
    if len(diff_tokens) > args.max_seq_length:
        diff_tokens = diff_tokens[:args.max_seq_length]
    else:
        diff_tokens += [src_vocab.stoi["<PAD>"]] * (args.max_seq_length - len(diff_tokens))
    
    diff_tensor = torch.tensor(diff_tokens, dtype=torch.long)
    
    # Generate commit message
    print("Generating commit message...")
    generated_message = generate_message(model, diff_tensor, tgt_vocab, device)
    
    print("\n" + "="*50)
    print("INPUT DIFF:")
    print(args.diff_text)
    print("\nGENERATED COMMIT MESSAGE:")
    print(generated_message)
    print("="*50)

if __name__ == "__main__":
    main()