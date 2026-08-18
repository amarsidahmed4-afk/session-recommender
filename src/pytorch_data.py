import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os

class SessionDataset(Dataset):
    def __init__(self, parquet_path, vocab_map, max_len=20):
        """
        Reads a Parquet chunk and maps product_ids to embedding indices (0 to N).
        """
        self.df = pd.read_parquet(parquet_path)
        self.vocab_map = vocab_map
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Map raw product_ids to our vocabulary indices
        # If an item isn't in vocab (rare), we use 0 (which we reserve for padding/unknown)
        raw_seq = row['input_sequence']
        mapped_seq = [self.vocab_map.get(item, 0) for item in raw_seq]
        target = self.vocab_map.get(row['target_item'], 0)
        
        # 2. Truncate if too long (keep the most recent items)
        if len(mapped_seq) > self.max_len:
            mapped_seq = mapped_seq[-self.max_len:]
            
        # 3. Pad with 0s if too short (pad on the left)
        # Example: [A, B] -> [0, 0, 0, A, B]
        pad_len = self.max_len - len(mapped_seq)
        padded_seq = [0] * pad_len + mapped_seq
        
        # Convert to PyTorch Tensors
        return torch.tensor(padded_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

def build_vocab(chunk_paths, max_vocab_size=100000):
    """
    Scans the data to find unique products and assigns each an index from 1 to N.
    Index 0 is strictly reserved for Padding.
    """
    print("Building vocabulary...")
    unique_items = set()
    
    for path in chunk_paths:
        df = pd.read_parquet(path, columns=['target_item'])
        unique_items.update(df['target_item'].unique())
        if len(unique_items) >= max_vocab_size:
            break
            
    # Index 0 is padding, so we start assigning at 1
    vocab_map = {item: idx + 1 for idx, item in enumerate(list(unique_items)[:max_vocab_size])}
    print(f"Vocabulary built with {len(vocab_map)} unique products.")
    return vocab_map

if __name__ == "__main__":
    # Quick local test to verify our logic!
    import glob
    chunks = glob.glob('data/features/sequence_chunks/*.parquet')
    
    # Build vocab from just the first chunk for a quick test
    vocab = build_vocab([chunks[0]])
    
    # Initialize Dataset and DataLoader
    dataset = SessionDataset(chunks[0], vocab_map=vocab, max_len=10)
    
    # DataLoader batches the data for us
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Grab one batch and inspect it
    x_batch, y_batch = next(iter(dataloader))
    
    print("\n--- Tensor Shapes ---")
    print(f"Input X shape: {x_batch.shape} (batch_size, max_len)")
    print(f"Target y shape: {y_batch.shape} (batch_size)")
    print("\n--- Example Input Tensor (X) ---")
    print(x_batch)