from dotenv import load_dotenv
import os

load_dotenv()

API_KEY_QDRANT = os.getenv("API_KEY_QDRANT")
API_KEY_RUNPOD = os.getenv("API_KEY_RUNPOD")
API_KEY_TMDB = os.getenv("API_KEY_TMDB")

headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {API_KEY_TMDB}"
}

DB_FILE = "movies.db"
TMDB_URL = "https://api.themoviedb.org/3/movie/{}"