from fastapi import APIRouter, Query
from app.services.rag import get_recommendations
from fastapi.responses import HTMLResponse
import re
router = APIRouter(prefix="/api")

@router.get("/recommend")
def recommend_movies(query: str = Query(...)):

    recommendations = get_recommendations(query)
    content = recommendations[0]["choices"][0]["tokens"][0]

    # Extract (title, description, poster) using regex
    matches = re.findall(
        r"Title Name:\s*(.*?)\nDescription:\s*(.*?)\nPoster:\s*(.*?\.jpg)",
        content,
        re.DOTALL,
    )

    html = "<h2 style='color: #fff;'>🎬 Recommended Movies:</h2><div style='display: flex; flex-wrap: wrap; gap: 20px;'>"

    for title, description, poster_path in matches:
        full_poster_url = f"https://image.tmdb.org/t/p/w342{poster_path}"
        html += f"""
        <div style="width: 220px; background: #1e1e1e; padding: 1rem; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.4); font-family: sans-serif;">
            <img src="{full_poster_url}" alt="{title}" style="width: 100%; border-radius: 10px;">
            <p style="color: #fff; font-weight: bold; margin: 0.5rem 0 0.25rem;">{title}</p>
            <p style="color: #ccc; font-size: 0.9rem;">{description.strip()}</p>
        </div>
        """

    html += "</div>"
    return HTMLResponse(content=html)
