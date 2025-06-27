# Necessary libraries
import requests # To make API requests
import pandas as pd
import os # For creating directories
from dotenv import load_dotenv # Securely loading API keys from a .env file
import time # For adding delay in case of retry

load_dotenv() # Loading environment variables from .env file

API_KEY = os.getenv("RAPIDAPI_KEY") # Fetching the RapidAPI key

url = "https://jsearch.p.rapidapi.com/search"

# Required headers
headers = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
}

# What job looking for and where
params = {
    "query": "computer science", # Keyword for the job title or description
    "location": "St. Louis, MO", # Target job location
    "page": 1, # Start from the first page
    "num_pages": 1 # Number of result pages to retrieve
}

# Function to fetch job data and save to CSV
def fetch_job_data():
    try:
        # Making the GET request to the API with headers and parameters
        response = requests.get(url, headers=headers, params=params)

        # Raising an exception if the request returened an error
        response.raise_for_status()

        # Parsse the response as JSON and extracting the 'data' part
        data = response.json()["data"]

        # Initializing a list to store each job posting
        job_list = []

        # Loop through each job in the result and store selected fields
        for job in data:
            job_list.append({
                "title": job.get("job_title", ""),
                "company": job.get("employer_name", ""),
                "location": job.get("job_city", "") + ", " + job.get("job_state", ""),
                "description": job.get("job_description", ""),
                "skills": ", ".join(job.get("job_required_skills", [])) if job.get("job_required_skills") else "N/A"
            })
        
        # Converting list of jobs into a DataFrame
        df = pd.DataFrame(job_list)

        # Making sure the 'data' folder exits
        os.makedirs("data", exist_ok=True)

        # Saving the DataFrame as a CSV file
        df.to_csv("data/job_postings.csv", index=False)
        print("Saved job data to data/job_postings.csv")
    
    except requests.exceptions.RequestException as e:
        # Printing any error that occured during the API request
        print("API Request Failed:", e)

# Running the function when this script is executed
if __name__ == "__main__":
    fetch_job_data()

