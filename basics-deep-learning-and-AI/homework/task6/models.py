import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import math
from tqdm import tqdm
import os
from .transformer_basics.transformer import get_pad_mask


ROOT = "./basics-deep-learning-and-AI/homework/task6/"


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_token_id: int = 0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.pad_token_id = pad_token_id
        self.max_len = max_len
        self.vocab_size = vocab_size

        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Transformer decoder layers
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)

        # Final projection layer
        self.projection = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Embed tokens and add positional encoding
        x = self.embedding(src)
        x = self.positional_encoding(x)

        # Generate causal mask if not provided
        if src_mask is None:
            src_mask = self._generate_square_subsequent_mask(src.size(1)).to(self.device)

        # Generate padding mask if not provided
        if src_key_padding_mask is None:
            src_key_padding_mask = (src == self.pad_token_id).to(self.device)

        # Pass through decoder
        x = self.decoder(
            x,
            memory=None,
            tgt_mask=src_mask,
            tgt_key_padding_mask=src_key_padding_mask,
            memory_key_padding_mask=None,
        )

        # Project to vocabulary size
        x = self.projection(x)

        return x

    def _generate_square_subsequent_mask(self, sz):
        """Generate a causal mask for the sequence."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def generate(self, prompt, max_length=200, temperature=1.0, top_k=50):
        """
        Generate text autoregressively.

        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated sequence
            temperature: Controls randomness (1.0 = normal, <1.0 = more conservative)
            top_k: Top-k sampling parameter (0 = no sampling)
        """
        self.eval()
        with torch.no_grad():
            # Tokenize prompt
            input_ids = self.tokenizer.encode(prompt).ids
            input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)

            generated = input_ids.clone()

            for _ in range(max_length):
                # Get predictions for last token
                outputs = self(input_ids[:, -self.max_len :])
                next_token_logits = outputs[0, -1, :] / temperature

                # Apply top-k filtering
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
                    min_val = top_k_values[-1]
                    next_token_logits[next_token_logits < min_val] = -float("Inf")

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to generated sequence
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

                # Stop if EOS token is generated
                if next_token.item() == self.tokenizer.token_to_id("</s>"):
                    break

            return self.tokenizer.decode(generated[0].tolist())


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt,
        memory=None,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        # Self attention
        tgt2 = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Feed forward
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])

    def forward(
        self,
        tgt,
        memory=None,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask)

        return output


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, "r", encoding="utf-8") as f:
            self.text = f.read()

        # Tokenize the entire text
        self.tokens = self.tokenizer.encode(self.text).ids

    def __len__(self):
        return len(self.tokens) // self.max_length

    def __getitem__(self, idx):
        start_idx = idx * self.max_length
        end_idx = start_idx + self.max_length

        # Get sequence of tokens
        tokens = self.tokens[start_idx:end_idx]

        # Input is all tokens except last, target is all tokens except first
        input_ids = tokens[:-1]
        target_ids = tokens[1:]

        # Pad sequences if needed
        if len(input_ids) < self.max_length - 1:
            pad_len = (self.max_length - 1) - len(input_ids)
            input_ids = input_ids + [self.tokenizer.token_to_id("<pad>")] * pad_len
            target_ids = target_ids + [self.tokenizer.token_to_id("<pad>")] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }


def train_model():
    # Initialize tokenizer
    tokenizer = Tokenizer.from_file(
        os.path.join(ROOT, "transformer_basics", "mistral_tokenizer.json")
    )

    # Create dataset and dataloader
    dataset = TextDataset(os.path.join(ROOT, "models.py"), tokenizer, max_length=128)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize model
    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        max_len=128,
        pad_token_id=tokenizer.token_to_id("<pad>"),
    )
    model.tokenizer = tokenizer  # Attach tokenizer to model for generation

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"))

    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids)

            # Calculate loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), f"transformer_epoch_{epoch+1}.pt")

    return model


if __name__ == "__main__":
    model = train_model()

    # Example generation
    prompt = "Once upon a time"
    generated_text = model.generate(prompt, max_length=100, temperature=0.8, top_k=50)
    print("\nGenerated text:")
    print(generated_text)
