
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import requests
import time
from sentence_transformers import SentenceTransformer
import os
from config.config import API_KEY_QDRANT, API_KEY_RUNPOD, headers, DB_FILE, TMDB_URL

qdrant = QdrantClient(
    url="https://02d9d3f7-4c63-4769-bf81-7c6cd1fcc9ce.eu-central-1-0.aws.cloud.qdrant.io",
    api_key=API_KEY_QDRANT
)

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

collection_name = "movies"


def retrieve_context(query, top_k=5):
    query_vec = embedding_model.encode(query).tolist()
    hits = qdrant.query_points(
        collection_name=collection_name,
        query=query_vec,
        limit=top_k,
        with_payload=True,
        score_threshold=0.1  # optional: filter weak matches
    ).points
    for hit in hits:
        print(f"Score: {hit.score} — {hit.payload['text']}")
    return [hit.payload["text"] for hit in hits]


def build_prompt(query, context_chunks):
    context = "\n\n".join(f"[MOVIE {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks))
    prompt = {
        "input": {
            "messages": [
                {
                    "role": "system",
                    "content": 
                        f"You are a movie recommendation assistant. Based on the given context:{context} of movie data"
                        "Do not generate or infer others."
                        "suggest movies that are similar to the one the user mentioned. "
                        "Use genres, plot similarities, and any other relevant details. "
                        "If no similar movie is found, say you don't know. "
                        "Always respond clearly and concisely in the same language as the question."
                        "Only recommend movies explicitly listed in the context. Do NOT mention any movies not present in the context." 
                        "For example context is:"
                        "Title name : Pirates of the Caribbean: The Curse of the Black Pearl Overview: After Port Royal is attacked and pillaged by a mysterious pirate crew, capturing the governor's daughter Elizabeth Swann in the process, William Turner asks free-willing pirate Jack Sparrow to help him locate the crew's ship—The Black Pearl—so that he can rescue the woman he loves. Poster path: safsdgds.jpg"
                        "Reccomendations should be in one line for one movie and have following format:"
                        "Title Name: Pirates of the Caribbean Description: After Port Royal is attacked and pillaged by a mysterious pirate crew, capturing the governor's daughter Elizabeth Swann in the process, William Turner asks free-willing pirate Jack Sparrow to help him locate the crew's ship—The Black Pearl—so that he can rescue the woman he loves. Poster: safsdgds.jpg"
                        "If context had more movies then do the same i next line of your answer."
                        
                        f"User question: {query}"
                    
                }
            ],
            "sampling_params": {
                "temperature": 0.2,
                "max_tokens": 300
            }
        }
    }

    return prompt

def call_vllm(prompt):

    RUNPOD_API_KEY = API_KEY_RUNPOD
    ENDPOINT_ID = 'w4sq1v7bd268di'

    base_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}

    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }

    submit_url = base_url + "run"


    print("Submitting request to vLLM...")
    submit_response = requests.post(submit_url, headers=headers, json=prompt)
    if submit_response.status_code != 200:
        raise Exception(f"Error submitting job: {submit_response.text}")
    
    job_id = submit_response.json()["id"]

    # Step 2: Poll for result
    status_url = base_url + f"status/{job_id}"
    print("Waiting for job to complete...")

    while True:
        status_response = requests.get(status_url, headers=headers)
        if status_response.status_code != 200:
            raise Exception(f"Error checking status: {status_response.text}")
        
        job = status_response.json()
        if job["status"] == "COMPLETED":
            return job["output"]
        elif job["status"] == "FAILED":
            raise Exception("Job failed.")
        
        time.sleep(2) 

def get_recommendations(query):  
    context = retrieve_context(query)
    prompt = build_prompt(query, context)
    result = call_vllm(prompt)
    return result