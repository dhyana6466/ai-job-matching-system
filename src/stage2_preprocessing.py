# Necessary Libraries
import pandas as pd
import re # For cleaning text using regular expressions
import os
import nltk # Natural Language Toolkit
from nltk.corpus import stopwords # List of common stopwords like 'the', 'is', etc
from nltk.stem import WordNetLemmatizer # For reducing words

# Downloading required resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initializing stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Loading the job postings dataset
df = pd.read_csv("data/job_postings.csv")

# Function to clean the job description and skills text
def clean_text(text):
    if pd.isnull(text): # If the text is missing, returning 'N/A'
        return "N/A"

    # Removing HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Removing special characters and punctuation
    text = re.sub(r"\s+", " ", text)

    # Converting all text to lowercase for consistency
    text = text.strip().lower()

    return text

# Function to remove stopwords and lemmatize the text
def lemmatize_text(text):
    words = text.split()

    # Removing stopwords and lemmatizing each word
    filtered = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(filtered)

# Cleaning and preprocessing description and skills columns
for col in ["description", "skills"]:
    df[col] = df[col].apply(clean_text) # Applying HTML + whitespace cleaning
    df[col] = df[col].apply(lemmatize_text) # Applying stopword removal + lemmatization

# Standardizing inconsistent location formatting manually
df["location"] = df["location"].replace({
    "saint louis, missouri": "st. louis, mo",
    "st louis, mo": "st. louis, mo"
})

# Filling any remaining missing values with 'N/A'
df.fillna("N/A", inplace=True)

# Making sure the 'data' folder exists
os.makedirs("data", exist_ok=True)

# Saving the cleaned dataset to a new CSV file
df.to_csv("data/job_postings_clean.csv", index=False)
print("Cleaned job data saved to data/job_postings_clean.csv")