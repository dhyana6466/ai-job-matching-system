# Necessary libraries
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Loading environment variables from .env file
load_dotenv()

# Initializing stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def compute_cosine_similarity(embedding1, embedding2):
    """
    Computes cosine similarity between two embeddings

    Parameters:
        embedding1 (list or numpy array): First vector
        embedding2 (list or numpy array): Second vector

    Returns:
        float: Similarity score 
    """
    return cosine_similarity([embedding1], [embedding2])[0][0]

def get_sentence_transformer_model(model_name="all-MiniLM-L6-v2"):
    """
    Loads a local sentence transformer model

    Parameters:
        model_name (str): Pre-trained model name

    Returns:
        SentenceTransformer: Loaded model object
    """
    return SentenceTransformer(model_name)

def clean_text(text):
    """
    Cleans the input text by removing HTML tags, special characters, and whitespace

    Parameters:
        text (str): Raw job descriptions or skills

    Returns:
        str: Cleaned lowercase string
    """
    if pd.isnull(text):
        return "N/A"
    text = re.sub(r"<.*?>", "", text) # Removing HTML tags
    text = re.sub(r"\s+", " ", text) # Removing excess whitespace
    return text.strip().lower() # Trim and lowercase

def lemmatize_text(text):
    """
    Removes stopwords and applies lemmatization

    Parameters:
        text (str): Cleaned text

    Returns:
        str: Processed text 
    """
    words = text.split()
    filtered = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(filtered)

def save_top_results(df_top10, output_path="data/top_10_similar_jobs.csv"):
    """
    Saves the top 10 job matches to a CSV file

    Parameters:
        df_top10 (DataFrame): DataFrame containing top job matches
        outputh_path (str): File path to save results

    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_top10.to_csv(output_path, index=False)
    print(f"\nTop10 similar jobs saved to {output_path}")

def load_resume_text(path="data/resume.txt"):
    """
    Loads the resume text file

    Parameters:
        path (str): Path to resume.txt
    
    Returns:
        str: Raw text content of resume
    """
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

