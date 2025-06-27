# Necessary Libraries
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Loading the job postings with preprocessed text
df = pd.read_csv("data/job_postings_clean.csv")

# Loading and reading the resume text
with open("data/resume.txt", "r", encoding="utf-8") as f:
    resume_text = f.read()

# Initializing the local embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Combining job description and skills into a single field
df["description"] = df["description"].fillna("")
df["skills"] = df["skills"].fillna("")
df["combined_text"] = df["description"] + " " + df["skills"]

# Ensuring all entries are strings
df["combined_text"] = df["combined_text"].apply(lambda x: str(x))

# Generating embeddings for all job descriptions
job_embeddings = model.encode(df["combined_text"].tolist(), convert_to_tensor=False)

# Generating embeddings for resume
resume_embedding = model.encode(resume_text, convert_to_tensor=False)

# Computing cosine similarity between resume and each job
similarity_scores = cosine_similarity([resume_embedding], job_embeddings)[0]
df["similarity"] = similarity_scores

# Getting top 10 job matches
top_jobs = df.sort_values(by="similarity", ascending=False).head(10).reset_index(drop=True)

# Printing the results
print("\n Top 10 Most Relevant Jobs Based on Resume Similarity:\n")
for i, row in top_jobs.iterrows():
    print(f"{i+1}. {row['title']} at {row['company']} ({row['location']})")
    print(f" Similarity Score: {row['similarity']: .4f}")
    print(" ---")

# Saving to CSV
os.makedirs("data", exist_ok=True)
top_jobs.to_csv("data/top_10_similar_jobs.csv", index=False)
print("\n Top 10 similar jobs saved to data/top_10_similar_jobs.csv")