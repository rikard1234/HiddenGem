import sqlite3
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

conn = sqlite3.connect("movies.db")

df = pd.read_sql_query("""
    SELECT id, title, overview, genres, vote_average, popularity, original_language, release_date, poster_path
    FROM movies
""", conn)

df = df[1:200]

qdrant = QdrantClient(
    url="https://02d9d3f7-4c63-4769-bf81-7c6cd1fcc9ce.eu-central-1-0.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.fp5NyoXY2E3YjGGTVoZP6MwtrnfSU73Mx3r4LjCSmvs"
)

qdrant.recreate_collection(
    collection_name="movies",
    vectors_config={"size": 384, "distance": "Cosine"} 
)

points = []
for _, row in df.iterrows():
    text = f"Title name : {row['title']} Overview: {row['overview']} Poster: {row['poster_path']}"
    vector = embedding_model.encode(text).tolist()
    points.append(
        PointStruct(
            id=int(row["id"]),
            vector=vector,
            payload={
                "text" : text,
                "db_id": int(row["id"]),
                "title": row["title"],
                "release_date": row["release_date"],
                "genres": row["genres"],
                "vote_average": row["vote_average"],
                "popularity": row["popularity"],
                "original_language": row["original_language"]
            }
        )
    )

BATCH_SIZE = 100

for i in range(0, len(points), BATCH_SIZE):
    qdrant.upsert(
        collection_name="movies",
        points=points[i:i + BATCH_SIZE]
    )



