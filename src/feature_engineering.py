import duckdb
import glob
import os

# Create directories
os.makedirs('tmp_duckdb', exist_ok=True)
os.makedirs('data/features/sequence_chunks', exist_ok=True)

# Find all our processed chunks
chunk_files = glob.glob('data/processed_nov/*.parquet')
print(f"Found {len(chunk_files)} chunks to process.")

# Initialize connection with safe limits
con = duckdb.connect()
con.execute("SET temp_directory='tmp_duckdb'")
con.execute("SET memory_limit='2GB'")
con.execute("SET threads=1") 

# Process each chunk sequentially (Batch Processing)
for i, file_path in enumerate(chunk_files):
    output_path = f'data/features/sequence_chunks/seq_chunk_{i}.parquet'
    print(f"Processing {file_path} -> {output_path} ...", end=" ")
    
    # We use a CTE to ensure the aliasing evaluates safely 
    query = f"""
    COPY (
        WITH grouped AS (
            SELECT
                user_session, 
                list(product_id ORDER BY event_time ASC) as ordered_product_ids,
                COUNT(*) as sequence_length
            FROM '{file_path}'
            GROUP BY user_session 
            HAVING COUNT(*) > 1
        )
        SELECT 
            user_session,
            sequence_length,
            ordered_product_ids[1:-2] as input_sequence,
            ordered_product_ids[-1] as target_item
        FROM grouped
    ) TO '{output_path}' (FORMAT parquet);
    """
    
    try:
        con.execute(query)
        print("Done!")
    except Exception as e:
        print(f"FAILED! Error: {e}")

print("All chunks processed successfully!")