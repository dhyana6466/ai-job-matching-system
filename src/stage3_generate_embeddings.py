# Necessary Libraries
import pandas as pd
import openai # OpenAI API for generating embeddings
import os
from dotenv import load_dotenv
import time

# Loading the environment variable from .env file
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = API_KEY # Setting the API key for the OpenAI

# Loading the cleaned job postings
df = pd.read_csv("data/job_postings_clean.csv")

# Combining description and skills for embedding input
df["combined_text"] = df["description"] + " " + df["skills"]

# Storing embeddings in a list
embeddings = []

# Loop through each job and generating embeddings
for idx, text in enumerate(df["combined_text"]):
    try:
        # Calling OpenAI embedding API
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )

        # Extracting embedding vector
        embedding_vector = response.data[0].embedding
        embeddings.append(embedding_vector)

        print(f"Embedded job {idx + 1}/{len(df)}")

        # Respecting rate limits with slight delat
        time.sleep(0.5)
    
    except Exception as e:
        print(f"Failed at index {idx}: {e}")
        embeddings.append([0.0] * 1536) # Placeholder vector if it fails

# Adding embeddings to the DataFrame
df["embedding"] = embeddings

# Saving the DataFrame with embeddings
os.makedirs("data", exist_ok=True)
df.to_pickle("data/job_postings_embedded.pkl") # Pickle for complex objects like lists

print("Job embeddings saved to data/job_postings_embedded.pkl")