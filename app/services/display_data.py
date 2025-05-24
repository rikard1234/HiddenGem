import pandas as pd
import sqlite3

conn = sqlite3.connect("movies.db")
#9000
# Load into a DataFrame
df = pd.read_sql_query("""
SELECT id, title, overview, genres, vote_average, popularity, original_language, release_date
FROM movies
""", conn)

conn.close()
print(df)