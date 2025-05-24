from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("API_KEY")
headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

DB_FILE = "movies.db"
TMDB_URL = "https://api.themoviedb.org/3/movie/{}"