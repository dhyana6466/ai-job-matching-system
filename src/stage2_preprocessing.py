# ---------------------------------------------
# Stage 2: Data Preprocessing - AI Job Matching
# Goal: Clean and prepare job data and resume for embedding
# Input: data/job_postings.csv, data/resume.txt
# Output: data/job_postings_clean.csv, data/resume_clean.txt
# ---------------------------------------------

# Necessary Libraries
import pandas as pd 
import re # For cleaning text using regular expressions
import os
import nltk # Natural Language Toolkit
from nltk.corpus import stopwords # List of common stopwords like 'the', 'is', etc
from nltk.stem import WordNetLemmatizer # Reducing words to their base form

# Downloading required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initializing stopwords set and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ----------------------------------------------------------
# Part 1: Cleaning and preprocessing job postings from CSV
# ----------------------------------------------------------

# Loading the job postings dataset from Stage 1
df = pd.read_csv("data/job_postings.csv")

# Function to clean text by removing HTML, extra spaces, and punctuation
def clean_text(text):
    if pd.isnull(text): # Handle missing text
        return "N/A"

    text = re.sub(r"<.*?>", "", text) # Removing HTML tags
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text) # Removing punctuation
    text = re.sub(r"\s+", " ", text) # Removing extra whitespace
    return text.strip().lower() # Converting to lowercase and trim

# Function to remove stopwords and lemmatize words
def lemmatize_text(text):
    words = text.split()
    filtered = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(filtered)

# Applying cleaning and preprocessing 'description' and 'skills' columns
for col in ["description", "skills"]:
    df[col] = df[col].apply(clean_text) 
    df[col] = df[col].apply(lemmatize_text)

# Standardizing location
df["location"] = df["location"].replace({
    "saint louis, missouri": "st. louis, mo",
    "st louis, mo": "st. louis, mo"
})

# Filling any remaining missing values
df.fillna("N/A", inplace=True)

# Ensuring the 'data' folder exists
os.makedirs("data", exist_ok=True)

# Saving the cleaned dataset to a new CSV file
df.to_csv("data/job_postings_clean.csv", index=False)
print("Cleaned job data saved to: data/job_postings_clean.csv")

# -----------------------------------------------------------
# Part 2: Cleaning and preprocessing the Students's resume
# -----------------------------------------------------------
resume_path = "data/resume.txt"

# Checking if resume.txt exists
if os.path.exists(resume_path):
    # Reading resume content
    with open(resume_path, "r", encoding="utf-8") as file:
        resume_text = file.read()
    
    # Cleaning and lemmatizing resume text
    cleaned_resume = clean_text(resume_text)
    lemmatized_resume = lemmatize_text(cleaned_resume)

    # Saving to new file
    with open("data/resume_clean.txt", "w", encoding="utf-8") as f:
        f.write(lemmatized_resume)

    print("Resume cleaned and saved to: data/resume_clean.txt")

else:
    print("resume.txt not found in data/folder.Please add it first")
