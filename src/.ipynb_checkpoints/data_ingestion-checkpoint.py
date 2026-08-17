import pandas as pd

# Safe data ingestion
# 1. Function to inspect data path
def inspect_sample(file_path, nrows=5):
    """
    Inspect the first few rows of a CSV file to understand its structure.

    Parameters:
    file_path (str): The path to the CSV file.
    nrows (int): Number of rows to read for inspection.

    Returns:
    DataFrame: A pandas DataFrame containing the first nrows of the CSV file.
    """
    try:
        sample_data = pd.read_csv(file_path, nrows=nrows)
        return sample_data
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None

# 2. Print the sample data and its .info()
print("Inspecting sample data from 'data/2019-Nov.csv':")
sample_data = inspect_sample('data/2019-Nov.csv')
print(sample_data)
print(sample_data.info())

# 3. Chunking data into parquet files
def csv_to_parquet_chunked(input_file, output_dir,
                           columns_keep=['event_time', 'event_type', 'product_id', 'category_id', 'user_session'], 
                           chunk_size=2000000):
    """
    Convert a large CSV file to Parquet format in chunks.

    Parameters:
    input_file (str): The path to the input CSV file.
    output_dir (str): The directory where the Parquet files will be saved.
    columns_keep (list): List of columns to keep.
    chunk_size (int): Number of rows per chunk.
    """
    try:
        for i, chunk in enumerate(pd.read_csv(input_file, usecols=columns_keep, chunksize=chunk_size)):
            # 1. Convert event_time to actual datetimes: 
            chunk['event_time'] = pd.to_datetime(chunk['event_time'])

            # 2. Convert event_type to a category: 
            chunk['event_type'] = chunk['event_type'].astype('category')

            # 3. Save the chunk to a Parquet file
            chunk.to_parquet(f"{output_dir}/chunk_{i}.parquet")
    except Exception as e:
        print(f"Error during conversion: {e}")

# 4. Chunking
print("Converting 'data/2019-Nov.csv' to Parquet format in chunks:")
csv_to_parquet_chunked('data/2019-Nov.csv', 'data/processed_nov')