# ---------------------------------------------
# Stage 1: Data Acquisition - AI Job Matching
# Goal: Fetch job postings using RapidAPI JSearch API
# Location focused: St. Louis, MO (can be changed)
# Filters applied: "computer science" jobs
# Output: data/job_postings.csv
# ---------------------------------------------

# Necessary libraries
import requests # To make HTTP requests to the API
import pandas as pd # To store and save data
import os # To manage folders
from dotenv import load_dotenv # To securely load API keys from .env file
import time # For request delay (to avoid rate limit)

# Loading the RapidAPI key from the .env file
load_dotenv() # This reads the .env file and loads the variables into the environment
API_KEY = os.getenv("RAPIDAPI_KEY") # The actual key is kept secret and never pushed to GitHub

# Setting up the API endpoint and headers
url = "https://jsearch.p.rapidapi.com/search"

headers = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
}

# A function to fetch job data using the API
def fetch_job_data(query="computer science", location="St. Louis, MO", pages=1):
    """
    Fetch job listings from the RapidAPI JSearch API and save to CSV

    Parameters:
        query (str): Job title or keyword to search for
        location (str): City and state for job search
        pages (int): Number of pages to retrieve
    """

    # List to store job results
    all_jobs = []

    for page in range(1, pages + 1):
        params = {
            "query": query,
            "location": location,
            "page": page,
            "num_pages": 1
        }
        
        try:
            # Making the API request
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status() # Raising an exception if the request returened an error
            
            # Extracting 'data' from response JSON
            data = response.json()["data"]
            
            # Extracting selected fields from each job
            for job in data:
                all_jobs.append({
                    "title": job.get("job_title", "N/A"),
                    "company": job.get("employer_name", "N/A"),
                    "location": job.get("job_city", "N/A") + ", " + job.get("job_state", "N/A"),
                    "description": job.get("job_description", "N/A"),
                    "skills": ", ".join(job.get("job_required_skills", [])) if job.get("job_required_skills") else "N/A"
                })

            print(f"Page {page} fetched successfully")
            time.sleep(1) # Respecting API rate limits
        
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch page {page}: {e}")
        
    # Converting list of jobs into a DataFrame
    df = pd.DataFrame(all_jobs)

    # Saving data to CSV
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/job_postings.csv", index=False)
    print(f"\nSaved {len(df)} job postings to data/job_postings.csv")

# Main execution
if __name__ == "__main__":
    # Can modify the query and location here
    fetch_job_data(query="computer science", location="St. Louis, MO", pages=2)

