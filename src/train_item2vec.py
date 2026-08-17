import pandas as pd
import glob
import os
from gensim.models import Word2Vec
from tqdm import tqdm

class SequenceGenerator:
    def __init__(self, chunk_pattern, limit_chunks=None):
        self.files = glob.glob(chunk_pattern)
        # DEV MODE: Only take the first N chunks to test the pipeline
        if limit_chunks:
            self.files = self.files[:limit_chunks]
            print(f"DEV MODE: Limited to {limit_chunks} chunk(s).")
        
    def __iter__(self):
        """
        Gensim will loop over this generator multiple times (vocab building + epochs).
        The progress bar will reset for each file in each epoch.
        """
        for file in self.files:
            df = pd.read_parquet(file)
            
            # Using tqdm to wrap the iterator gives us a live progress bar!
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Reading {os.path.basename(file)}"):
                # Reconstruct full sequence
                full_sequence = list(row['input_sequence']) + [row['target_item']]
                str_sequence = [str(item) for item in full_sequence]
                
                yield str_sequence

if __name__ == "__main__":
    os.makedirs('data/models', exist_ok=True)
    
    print("Initializing streaming generator...")
    # Limiting to 2 chunks for our local sanity check
    sentences = SequenceGenerator('data/features/sequence_chunks/*.parquet', limit_chunks=2)
    
    print("Training Item2Vec model...")
    # epochs=1 forces a single training pass. workers=1 is safe for memory.
    model = Word2Vec(
        sentences=sentences, 
        vector_size=64, 
        window=5, 
        min_count=2, 
        workers=1, 
        epochs=1 
    )
    
    model.save("data/models/item2vec.model")
    print("\nModel saved successfully to data/models/item2vec.model!")