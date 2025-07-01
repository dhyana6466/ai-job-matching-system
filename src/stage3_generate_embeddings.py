# ---------------------------------------------
# Stage 3: Embedding Information - AI Job Matching
# Goal: Convert job description + skills into numerical embeddings using SentenceTransformer
# Model Used: all-MiniLM-L6-v2 (Free, local model)
# Input: data/job_postings_clean.csv
# Output: data/job_postings_embedded.pkl
# ---------------------------------------------

# Necessary Libraries
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

# Loading the cleaned job postings dataset from Stage 2
df = pd.read_csv("data/job_postings_clean.csv")

# Combining job description and skills into one field for embedding
df["combined_text"] = df["description"].astype(str) + " " + df["skills"].astype(str)

# Loading the local sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2") # Fast and effective for semantic similarity

# Generating embeddings for each combined text
print("Generating embeddings for job postings...")
df["embedding"] = df["combined_text"].apply(lambda x: model.encode(str(x)).tolist())

# Saving the DataFrame with embeddings
os.makedirs("data", exist_ok=True)
df.to_pickle("data/job_postings_embedded.pkl") # Pickle for complex objects like lists

print("Job embeddings saved to: data/job_postings_embedded.pkl")