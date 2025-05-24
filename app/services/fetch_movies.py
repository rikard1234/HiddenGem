import json
import sqlite3
import requests
from time import sleep
from tqdm import tqdm
import os 
from dotenv import load_dotenv

conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS movies (
    id INTEGER PRIMARY KEY,
    title TEXT,
    overview TEXT,
    release_date TEXT,
    genres TEXT,
    original_language TEXT,
    vote_average REAL,
    vote_count INTEGER,
    popularity REAL,
    poster_path TEXT,
    backdrop_path TEXT
);
""")
conn.commit()

with open("movie_ids_05_15_2025.json", "r", encoding="utf-8") as f:
    movie_ids = [json.loads(line)["id"] for line in f if line.strip()]
print(movie_ids[:10])

for movie_id in tqdm(movie_ids[7700:15000]):  # Adjust limit as needed
    try:
        url = TMDB_URL.format(movie_id)
        response = requests.get(url, headers=headers)
        print(response.text)
        if response.status_code != 200:
            sleep(0.25)  # avoid rate limits
            continue
        data = response.json()
        genres = ",".join([g["name"] for g in data.get("genres", [])])

        cursor.execute("""
        INSERT OR REPLACE INTO movies (
            id, title, overview, release_date, genres, original_language,
            vote_average, vote_count, popularity, poster_path, backdrop_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["id"],
            data.get("title", ""),
            data.get("overview", ""),
            data.get("release_date", ""),
            genres,
            data.get("original_language", ""),
            data.get("vote_average", 0.0),
            data.get("vote_count", 0),
            data.get("popularity", 0.0),
            data.get("poster_path", ""),
            data.get("backdrop_path", "")
        ))
        conn.commit()
        print(f"Inserting: {data.get('title')} ({data['id']})")
        sleep(0.25)  

    except Exception as e:
        print(f"Error fetching {movie_id}: {e}")
        continue

# --- Done ---
conn.close()
print("✅ Done storing movie data.")