import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from groq import Groq

# Laden des gespeicherten FAISS-Index und der Chunk-Daten
with open("faiss/chunks_mapping.pkl", "rb") as f:
    all_chunks = pickle.load(f)

index = faiss.read_index("faiss/faiss_index.index")

# Embedding-Modell laden
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Prompt-Builder
def build_prompt(context_chunks, user_query):
    context_block = "\n\n".join(context_chunks)
    prompt = f"""Answer the following question based on the given context.

context:
{context_block}

query:
{user_query}

answer:"""
    return prompt

# Retrieval
def retrieve(query, k=10):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    retrieved_texts = [all_chunks[i]["text"] for i in indices[0]]
    return retrieved_texts, None, distances[0]

# Validierung
def retrieve_with_validation(user_query, k=3, threshold=30.0):
    chunks, _, distances = retrieve(user_query, k=k)
    if distances is None or len(distances) == 0:
        return [], False

    max_distance = np.max(distances[0]) if isinstance(distances, np.ndarray) and distances.ndim == 2 else np.max(distances)
    if max_distance > threshold:
        return [], False
    return chunks, True

# API-Key laden
load_dotenv(dotenv_path="env/.env")
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("API Key not found! Please check env/.env")

client = Groq(api_key=groq_api_key)