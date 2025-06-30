# ---------------------------------------------
# Stage 4: Similarity Calculation & Top Job Selection - AI Job Matching
# Goal: Compare resume with job postings using cosine similarity and extract top matches
# Input: data/job_postings_clean.csv, data/resume.txt
# Output: data/top_10_similar_jobs.csv
# ---------------------------------------------

# Necessary Libraries
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Loading cleaned data from Stage 2
# Loading the cleaned job postings
df = pd.read_csv("data/job_postings_clean.csv")

# Loading the pre-cleaned resume text
resume_path = "data/resume_clean.txt"
if os.path.exists(resume_path):
    with open(resume_path, "r", encoding="utf-8") as f:
        resume_text = f.read()
else:
    raise FileNotFoundError("Cleaned resume not found. Please run Stage 2 preprocessing")

# Initializing Sentence Embedding Model
# Using a local SentenceTranformer model to avoid API limits
model = SentenceTransformer('all-MiniLM-L6-v2')

# Combining 'description' and 'skills' into one text field for embedding
df["description"] = df["description"].fillna("")
df["skills"] = df["skills"].fillna("")
df["combined_text"] = df["description"] + " " + df["skills"]

# Ensuring all entries are strings
df["combined_text"] = df["combined_text"].astype(str)

# Generating embeddings for jobs and resume
print("Generating embeddings for job postings and resume...")
# Generating embeddings for all job postings
job_embeddings = model.encode(df["combined_text"].tolist(), convert_to_tensor=False)

# Generating embeddings for resume
resume_embedding = model.encode(resume_text, convert_to_tensor=False)

# Computing Similarity Scores
# Computing cosine similarity between resume and each job description
similarity_scores = cosine_similarity([resume_embedding], job_embeddings)[0]

# Adding similarity scores to the dataframe
df["similarity"] = similarity_scores

# Selecting top 10 most relevant job matches
# Sorting job postings by similarity (highest first)
top_jobs = df.sort_values(by="similarity", ascending=False).head(10).reset_index(drop=True)

# Printing results
print("\n Top 10 Most Relevant Jobs Based on Resume Similarity:\n")
for i, row in top_jobs.iterrows():
    print(f"{i+1}. {row['title']} at {row['company']} ({row['location']})")
    print(f" Similarity Score: {row['similarity']: .4f}")
    print(" ---")

# Saving results
os.makedirs("data", exist_ok=True)
top_jobs.to_csv("data/top_10_similar_jobs.csv", index=False)
print("\n Top 10 similar jobs saved to data/top_10_similar_jobs.csv")

# Visualizing similarity scores as bar chart
import matplotlib.pyplot as plt

# Plotting similarity scores for top 10 jobs
plt.figure(figsize=(10, 6))
plt.barh(top_jobs["title"] + " at " + top_jobs["company"], top_jobs["similarity"], color='skyblue')
plt.xlabel("Cosine Similarity Score")
plt.title("Top 10 Most Relevant Job Matches Based on Resume")
plt.gca().invert_yaxis() # Highest score at the top
plt.tight_layout()
plt.savefig("data/top_10_similarity_plot.png")
plt.show()

print("Bar chart saved as: data/top_10_similarity_plot.png")