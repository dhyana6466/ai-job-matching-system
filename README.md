# AI-Powered Job Matching System
This project uses AI to match a user's resume with the most relevant job descriptions using semantic embedding and cosine similarity. The system is built to be lighweight, fast, and fully client-operable using local embeddings instead of relying on cloud APIs.

---

## Features
- Local resume-to-job matching using NLP
- Embeddings generated via `sentence-transformers`
- Cosine similarity ranking to find the best job matches
- Saves top 10 job results to a  CSV
- Fully commented and modular Python code

---

## Project Structure
```
ai-job-matching-system/
│
├── data/
│ ├── job_postings_clean.csv # Cleaned job listings
│ ├── job_postings_embedded.pkl # Pickled job embeddings
│ ├── resume.txt # User's resume in plain text
│ └── top_10_similar_jobs.csv # Final output: top 10 matches
│
├── src/
│ ├── stage1_data_acquisition.py # Collect data via API
│ ├── stage2_preprocessing.py # Clean and format job data
│ ├── stage3_generate_embeddings.py # Generate job embeddings
│ ├── stage4_similarity_and_ranking.py # Match resume to jobs
│ └── utils.py # Helper functions
|
├── .env
├── README.md
├── report.md
└── requirements.txt
```

## How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/dhyana6466/ai-job-matching-system 
cd ai-job-matching-system

### 2. Install depenencies
pip install -r requirements.txt

### 3. Add resume
Put the plain text resume in the data/ directory as resume.txt

### 4. Run the stages
# Collecting data via API
python src/stage1_data_acquisition.py

# Preprocess job data
python src/stage2_preprocessing.py

# Generate embeddings
python src/stage3_generate_embeddings.py

# Match jobs and calculate similarity
python src/stage4_similarity_and_ranking.py
```

## Output
You'll get the top 10 most relevant job matches in:
data/top_10_similary_jobs.csv
