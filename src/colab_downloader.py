import kagglehub
import os
import shutil

def setup_and_download():
    """
    Authenticates with Kaggle via OAuth and downloads the dataset directly
    to the Colab server's high-speed storage.
    """
    print("Initiating Kaggle authentication...")
    # In Colab, this will print a URL. You click it, log in, and paste the code back.
    kagglehub.login()
    
    print("\nDownloading dataset (using Colab's gigabit internet)...")
    path = kagglehub.dataset_download("mkechinov/ecommerce-behavior-data-from-multi-category-store")
    print(f"Dataset downloaded to Kaggle cache: {path}")
    
    os.makedirs('data', exist_ok=True)
    csv_path = os.path.join(path, "2019-Nov.csv")
    target_path = "data/2019-Nov.csv"
    
    if os.path.exists(csv_path):
        print(f"Copying {csv_path} to {target_path}...")
        # We copy instead of move to avoid corrupting the kagglehub cache
        shutil.copy(csv_path, target_path)
        print("Data setup complete! Ready for data_ingestion.py")
    else:
        print("Error: 2019-Nov.csv not found in the downloaded folder.")

if __name__ == "__main__":
    setup_and_download()