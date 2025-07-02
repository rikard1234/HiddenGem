from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict, List
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
import requests
import time
from config.config import API_KEY_QDRANT, API_KEY_RUNPOD
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings
import asyncio
import httpx

class RunPodLLM(Runnable):
    def __init__(self, endpoint_id: str, api_key: str):
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}/"
        self.submit_url = self.base_url + "run"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def invoke(self, inputs, config=None, **kwargs) -> str:
        if hasattr(inputs, "to_messages"): 
            messages = inputs.to_messages()
            formatted_messages = [
                {"role": m.type, "content": m.content} for m in messages
            ]
        elif isinstance(inputs, dict):
            input_text = inputs.get("input") or inputs.get("prompt")
            if input_text is None:
                raise ValueError("Expected 'input' or 'prompt' in dict")
            formatted_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ]
        else:
            formatted_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": str(inputs)}
            ]

        payload = {
            "input": {
                "messages": formatted_messages,
                "sampling_params": {
                    "temperature": 0.2,
                    "max_tokens": 300
                }
            }
        }

        print("Submitting request to RunPod...")
        submit_response = requests.post(self.submit_url, headers=self.headers, json=payload)
        if submit_response.status_code != 200:
            raise Exception(f"RunPod error (submit): {submit_response.text}")

        job_id = submit_response.json()["id"]
        status_url = self.base_url + f"status/{job_id}"
        print(f"Job submitted. Job ID: {job_id}")

        # Step 3: Poll until the job is done
        print("Waiting for job to complete...")
        while True:
            status_response = requests.get(status_url, headers=self.headers)
            if status_response.status_code != 200:
                raise Exception(f"RunPod error (status): {status_response.text}")

            job_data = status_response.json()
            if job_data["status"] == "COMPLETED":
                output = job_data["output"]
                try:
                    tokens = output[0]["choices"][0]["tokens"]
                    return "".join(tokens).strip()
                except Exception as e:
                    raise Exception(f"Unexpected RunPod output format: {e}")
            elif job_data["status"] == "FAILED":
                raise Exception("RunPod job failed.")

            time.sleep(1)

    async def ainvoke(self, inputs, config=None, **kwargs) -> str:
        if hasattr(inputs, "to_messages"): 
            messages = inputs.to_messages()
            formatted_messages = [
                {"role": m.type, "content": m.content} for m in messages
            ]
        elif isinstance(inputs, dict):
            input_text = inputs.get("input") or inputs.get("prompt")
            if input_text is None:
                raise ValueError("Expected 'input' or 'prompt' in dict")
            formatted_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ]
        else:
            formatted_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": str(inputs)}
            ]

        payload = {
            "input": {
                "messages": formatted_messages,
                "sampling_params": {
                    "temperature": 0.2,
                    "max_tokens": 300
                }
            }
        }

        async with httpx.AsyncClient() as client:
            print("Submitting request to RunPod...")
            submit_response = await client.post(self.submit_url, headers=self.headers, json=payload)
            if submit_response.status_code != 200:
                raise Exception(f"RunPod error (submit): {submit_response.text}")

            job_id = submit_response.json()["id"]
            status_url = self.base_url + f"status/{job_id}"
            print(f"Job submitted. Job ID: {job_id}")

            print("Waiting for job to complete...")
            while True:
                status_response = await client.get(status_url, headers=self.headers)
                if status_response.status_code != 200:
                    raise Exception(f"RunPod error (status): {status_response.text}")

                job_data = status_response.json()
                if job_data["status"] == "COMPLETED":
                    output = job_data["output"]
                    try:
                        tokens = output[0]["choices"][0]["tokens"]
                        return "".join(tokens).strip()
                    except Exception as e:
                        raise Exception(f"Unexpected RunPod output format: {e}")
                elif job_data["status"] == "FAILED":
                    raise Exception("RunPod job failed.")

                await asyncio.sleep(1)


system_prompt = (
    "You are a movie recommendation assistant. Based on the given context:{context} of movie data"
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
)


qdrant = QdrantClient(
    url="https://02d9d3f7-4c63-4769-bf81-7c6cd1fcc9ce.eu-central-1-0.aws.cloud.qdrant.io",
    api_key=API_KEY_QDRANT
)

embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

collection_name = "movies"

vector_store = Qdrant(
    client=qdrant,
    collection_name=collection_name,
    embeddings=embedding_model,
    content_payload_key="text",
    metadata_payload_key="metadata"
)

retriever = vector_store.as_retriever()
vector_store.asimilarity_search
llm = RunPodLLM(endpoint_id='w4sq1v7bd268di', api_key=API_KEY_RUNPOD)



class State(TypedDict):
    input: str
    context: List[Document]
    answer: str
    sources: List[str]

# Retrieval step
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["input"])
    return {"context": retrieved_docs}

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Question: {input}\n\nContext:\n{context}")
])

# Generation step
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({
        "input": state["input"],
        "context": docs_content
    })
    response = llm.invoke(messages)
    sources = []
    for doc in state["context"]:
        db_id = doc.metadata.get("title")
        if db_id is not None:
            sources.append(str(db_id))  # convert to str if needed
    
    return {
        "answer": response,
        "sources": sources
    }

# Compile graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Run graph
response = graph.invoke({"input": "pirates"})

print(response["answer"])
print(f'Sources: {response["sources"]}')