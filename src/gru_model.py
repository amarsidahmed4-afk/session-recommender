import torch
import torch.nn as nn

class SessionGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        """
        vocab_size: Total number of unique products + 1 (for padding)
        embedding_dim: Size of the item vectors (like Item2Vec)
        hidden_dim: The size of the GRU's "memory" state
        """
        super(SessionGRU, self).__init__()
        
        # 1. Embedding Layer: Turns integer IDs into dense vectors
        # padding_idx=0 is CRITICAL: it tells PyTorch not to calculate gradients for the 0s
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embedding_dim, 
            padding_idx=0
        )
        
        # 2. GRU Layer: Processes the sequence of vectors
        self.gru = nn.GRU(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            batch_first=True
        )
        
        # 3. Output Layer: Maps the final hidden state back to vocab probabilities
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        # x shape: (batch_size, max_len)
        
        # Embed the items
        embedded = self.embedding(x) 
        # embedded shape: (batch_size, max_len, embedding_dim)
        
        # Pass through GRU
        gru_out, hidden = self.gru(embedded)
        # hidden shape: (num_layers, batch_size, hidden_dim) -> represents the FINAL state
        
        # We only care about the final hidden state to predict the *next* item
        # Squeeze removes the num_layers dimension (since we only have 1 layer)
        final_hidden = hidden.squeeze(0) 
        # final_hidden shape: (batch_size, hidden_dim)
        
        # Pass through linear layer to get raw unnormalized predictions (logits) for every item
        logits = self.fc(final_hidden) 
        # logits shape: (batch_size, vocab_size)
        
        return logits

if __name__ == "__main__":
    # Quick local sanity check!
    # Let's pretend we have a small vocabulary of 10,000 items
    test_vocab_size = 10000
    model = SessionGRU(vocab_size=test_vocab_size)
    
    # Dummy batch of 4 sequences, padded to length 10 (just like your dataset output!)
    dummy_x = torch.randint(0, test_vocab_size, (4, 10))
    
    # Pass it through the model
    logits = model(dummy_x)
    
    print("\n--- GRU Architecture Sanity Check ---")
    print(f"Input shape X: {dummy_x.shape} (batch_size, max_len)")
    print(f"Output shape Logits: {logits.shape} (batch_size, vocab_size)")
    print("Success: Model initialized and forward pass is fully functional!")