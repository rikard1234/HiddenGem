
import requests
url = "https://api.themoviedb.org/3/authentication"

headers = {
    "accept": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJhNTE5OGMwNzFlMWY1M2Q0ZDdiYTgzZjRlYWVlODU4YSIsIm5iZiI6MTc0Njk4NTgyOS40NzEsInN1YiI6IjY4MjBlMzY1YWZjNzM1MThjNDczOGYyZiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.fdmQF2Fz_GVv2TtEPTqeGIY6D5yQkKBBk8-JSkNjnps"
}

response = requests.get(url, headers=headers)

print(response.text)