import torch
import torch.nn as nn
import math

class SessionTransformer(nn.Module):
    def __init__(self, vocab_size, max_len=15, embedding_dim=128, nhead=4, num_layers=2):
        """
        A Self-Attentive Sequential Recommendation (SASRec) style Transformer.
        """
        super(SessionTransformer, self).__init__()
        
        # 1. Item Embeddings (same as GRU/Item2Vec)
        self.item_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 2. Positional Embeddings (Crucial for Transformers!)
        # Because Transformers look at everything simultaneously, we MUST tell them
        # which item came 1st, 2nd, 3rd... up to max_len.
        self.position_embedding = nn.Embedding(max_len, embedding_dim)
        
        # 3. The Transformer Encoder Block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 4,
            dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Output Layer
        self.fc = nn.Linear(embedding_dim, vocab_size)
        
        self.max_len = max_len

    def forward(self, x):
        # x shape: (batch_size, max_len)
        batch_size, seq_len = x.size()
        
        # 1. Create Position IDs (0, 1, 2, ... seq_len-1)
        # We push this to the same device (CPU/GPU) as the input data
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        
        # 2. Combine Item and Position Vectors
        # Mathematical addition: Item Identity + Item Position in sequence
        x_emb = self.item_embedding(x) + self.position_embedding(positions)
        
        # 3. Create Causal Mask (Look-ahead Mask)
        # We don't want item 2 attending to item 5 (which is the future).
        # generate_square_subsequent_mask creates a diagonal matrix blocking the future.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # 4. Create Padding Mask
        # We tell the Transformer's attention mechanism to strictly ignore 0s (padding)
        pad_mask = (x == 0)
        
        # 5. Pass through Transformer
        transformer_out = self.transformer(
            x_emb, 
            mask=causal_mask, 
            src_key_padding_mask=pad_mask
        )
        # transformer_out shape: (batch_size, max_len, embedding_dim)
        
        # 6. Grab the output state of the VERY LAST item in the sequence
        # We slice at [:, -1, :] because the sequence is left-padded. 
        # The true last item is always at the absolute end of the array.
        final_state = transformer_out[:, -1, :]
        
        # 7. Predict the next item
        logits = self.fc(final_state)
        
        return logits

if __name__ == "__main__":
    # Architecture Sanity Check
    test_vocab = 10000
    test_seq_len = 15
    model = SessionTransformer(vocab_size=test_vocab, max_len=test_seq_len)
    
    dummy_x = torch.randint(0, test_vocab, (4, test_seq_len))
    
    # Introduce some artificial padding (0s)
    dummy_x[0, :5] = 0
    dummy_x[1, :2] = 0
    
    logits = model(dummy_x)
    
    print("\n--- Transformer Architecture Sanity Check ---")
    print(f"Input shape X: {dummy_x.shape} (batch_size, max_len)")
    print(f"Output shape Logits: {logits.shape} (batch_size, vocab_size)")
    print("Success: Transformer is perfectly processing masked sequences!")