import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import glob
from tqdm import tqdm
import os
import re

from pytorch_data import SessionDataset, build_vocab
from gru_model import SessionTransformer

def get_chunk_num(path):
    """Extracts the integer chunk number for proper chronological sorting."""
    match = re.search(r'seq_chunk_(\d+)', path)
    return int(match.group(1)) if match else -1

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1. Production Data Setup (Strict Chronological Sorting)
    raw_paths = glob.glob('data/features/sequence_chunks/*.parquet')
    chunk_paths = sorted(raw_paths, key=get_chunk_num)
    
    print("Building vocabulary from the first 5 chunks (to save time)...")
    vocab_map = build_vocab(chunk_paths[:5], max_vocab_size=50000)
    vocab_size = len(vocab_map) + 1 
    
    # Train on the past, validate on the absolute future (Last chunk)
    train_chunks = chunk_paths[:-1]
    val_chunk = chunk_paths[-1]

    # 2. Initialize Model
    model = SessionTransformer(vocab_size=vocab_size, embedding_dim=128, hidden_dim=256)
    model = model.to(device)

    # 3. Loss and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Production Training Loop
    epochs = 2
    os.makedirs("data/models/checkpoints", exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        
        for chunk_idx, chunk_path in enumerate(train_chunks):
            train_dataset = SessionDataset(chunk_path, vocab_map, max_len=15)
            train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True) 
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} | Chunk {chunk_idx+1}/{len(train_chunks)}")
            
            # TELEMETRY FIX: Initialize running loss for this chunk
            running_loss = 0.0
            
            for step, (x_batch, y_batch) in enumerate(progress_bar, 1):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad() 
                logits = model(x_batch) 
                
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                
                # Accumulate loss and calculate rolling average
                running_loss += loss.item()
                avg_loss = running_loss / step
                
                # Log the smooth average loss, preventing micro-batch spikes
                progress_bar.set_postfix({'avg_loss': f"{avg_loss:.4f}"})
        
        # 5. Model Checkpointing
        checkpoint_path = f"data/models/checkpoints/gru_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"\n[*] Model weights safely backed up to: {checkpoint_path}")
        
        # 6. Validation (Test on the final, true future chunk)
        print(f"\nRunning Validation on Future Data ({val_chunk})...")
        model.eval() 
        correct = 0
        total = 0
        
        val_dataset = SessionDataset(val_chunk, vocab_map, max_len=15)
        val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
        
        with torch.no_grad(): 
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for x_batch, y_batch in val_bar:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                
                predictions = torch.argmax(logits, dim=1)
                mask = (y_batch != 0)
                correct += ((predictions == y_batch) & mask).sum().item()
                total += mask.sum().item()
        
        val_accuracy = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1} Summary - Val Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)\n")

if __name__ == "__main__":
    train_model()