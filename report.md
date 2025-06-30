# AI-Powered Job Matching System: Project Report

## Overview
This project compares a student resume with job descriptions using sentence embeddings. Then it givens the top 10 job recommendations that are most similar to the resume.

---

## Stage 1: Data Acquisitions
We used the **JSearch API** to programmatically fetch job posting based on filters such as location, keyword (e.g., "computer science), and employment type. This approach ensured we had relevant and real-world job descriptions in structured JSON format.

**Tools Used:**
- Python `requests` library for API calls
- JSeach API (RapidAPI) for job listings

**Challenges Faced:**
- API rate limits required handling pagination and retires
- Some job descriptions were incomplete or incosistent across listings

---

## Stage 2: Data Preprocessing
Extracted and combined key fields such as:
- `job_title`
- `company_name`
- `location`
- `job_description`
- `skills`

**Processing steps included:**
- Lowercasing text
- Removing missing values
- Merging descriptions and skills into a `combined_text` column for embedding

Saved as `data/job_postings_clean.csv`

---

## Stage 3: Embedding Generation
Used the **Sentence-BERT model (`all-MiniLM-L6-v2`)** from the `sentence-transformers` library to convert job descriptions and the resume into embeddings (high-dimensional vectors)

**Why Sentence-BERT>**
- Fast and lighweight
- Pre-trained for semnatic similarity tasks
- Works well with limited computational resources

Resume text was loaded from `data/resume.txt`, cleaned, and encoded to produce a embedding vector

---

## Stage 4: Similarity Computation & Ranking
Used **cosine similarity** to compare the resume vector with each job posting vector. The similarity score indicates how closely a job matches the resume (1 = perfect match, 0 = no similarity)

**Steps**
1. Compute cosine similarity between resume embedding and each job embedding
2. Rank jobs in descending order of similarity
3. Display top 10 most relevant jobs with company, location, and score

Saved results to `data/top_10_similar_jobs.csv`
Similarity plot: `data/top_10_similarity_plot.png`

---

## Results & Evaluation

### Top 10 Relevant Jobs (example):
1. Computer Science Tutor at BuffTutor (0.4554)
2. DARPA Computer Engineer at Blue Sky (0.4518)
3. Computer Scientist at DoD (0.4502)
4. ...

### Strengths:
- Local embeddings removed the depedency on OpenAI API
- Job relevance was semantically meaningful, not keyword-based

## Weaknesses:
- Resume parsing was basic (plain text only)
- Cosine similarity doesn't capture all context (e.g., seniority)

---

### Future Improvements
- Use a more sophisticated resume parser (like `pyresparser` or `spaCY`)
- Incorporate filters like job location preference, salary range
- Use classification or clustering to pre-group job types
- Add a web interface for uploading resumes and viewing matched jobs

---

## Summary
This project demonstrates how **Natural Language Processing (NLP)** and **vector embeddings** can be used to build a smart job recommendation system. By matching a resume's semantic content with real job descriptions, the system enables personalized and relevant job search experiences.