# Real-Time E-Commerce Session Recommender

## The Business Problem
In e-commerce, the vast majority of users browse anonymously, rendering traditional matrix-factorization recommendation systems (like Collaborative Filtering) useless. This project implements a Session-Based Recommendation System that predicts a user's next click purely based on their short-term chronological actions.

## The Architecture & Constraints
This project was initially engineered under strict local hardware constraints (2GB RAM Chromebook), requiring a highly optimized out-of-core processing pipeline before scaling to Cloud GPUs.
* Hardware Constraint & Out-of-Core Processing:Engineered a horizontal batch-processing pipeline using DuckDB to process 67 million rows locally without crashing.
* Storage Optimization: Transitioned 8.4 GB of raw text logs into highly compressed, columnar Parquet files, reducing storage by 77%.
* Cloud Migration: Built modular src/ scripts to allow one-click Infrastructure-as-Code (IaC) deployment to Google Colab (NVIDIA T4 GPUs) for deep learning.

## Modeling Progression
1. **Heuristic Baseline (SQL):** Established a 48.5% accuracy baseline by writing a DuckDB rule to simply predict the last-viewed item.
2. **Markov Chain:** Built a global Transition Matrix in SQL, capturing first-order relationships and achieving 49.0% accuracy.
3. **Item2Vec Embeddings:** Stream-trained a 64-dimensional Word2Vec architecture to capture latent semantic product similarities using gensim generators to prevent RAM overflow.
4. **Deep Learning (PyTorch GRU):**
* Engineered custom PyTorch DataLoaders implementing Left-Padding to handle variable-length anonymous sessions.
* Caught and resolved a critical Data Leakage bug caused by lexicographical sorting (chunk_10 before chunk_2), enforcing strict chronological train/validation splits.
* The pure GRU achieved 43.8% accuracy, failing to beat the heuristic baseline due to the model struggling with immediate repetition patterns (e.g., page refreshes) and Black Friday concept drift.
5. **Transformers (SASRec):** Formulated a Self-Attentive Sequential Recommendation architecture utilizing Look-ahead (Causal) masking and Positional Encoding to overcome the GRU's hidden-state bottleneck.

## Tech Stack
* **Data Engineering:** DuckDB, Pandas, PyArrow, Parquet
* **Machine Learning:** PyTorch, Gensim (Word2Vec)
* **MLOps:** Batch Checkpointing, Out-of-Core Computation, Infrastructure-as-Code.

--- 

## How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Data Pipeline (Will download data via Kaggle API)
python src/colab_downloader.py
python src/data_ingestion.py
python src/feature_engineering.py

# 3. Train Models
python src/train_item2vec.py
python src/train_gru.py
```