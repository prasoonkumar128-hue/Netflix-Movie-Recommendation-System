
# 🎬 Movie Recommendation System

## Overview
A content-based movie recommendation system that suggests five similar movies based on movie descriptions and genres.

## Features
- Content-based movie recommendations
- CountVectorizer and TF-IDF comparison
- Cosine similarity for ranking movies
- Movie posters, release years and ratings using TMDB
- Interactive Streamlit web interface

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- TMDB API

## How to Run

1. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

2. Configure your TMDB API key in `.streamlit/secrets.toml`:

   ```toml
   TMDB_API_KEY = "YOUR_TMDB_API_KEY"
   ```

3. Run the application:

   ```bash
   python -m streamlit run app.py
   ```

## Project Files
- `app.py` — Streamlit application
- `Main.ipynb` — Model development and evaluation
- `dataset.csv` — Movie dataset
- `movies_list.pkl` — Processed movie data
- `similarity.pkl` — Final TF-IDF similarity matrix
- `requirements.txt` — Required Python libraries

## Recommendation Method
The system uses TF-IDF to convert movie text into numerical vectors and cosine similarity to identify movies with similar content.
