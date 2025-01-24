from fastapi import FastAPI
from pydantic import BaseModel
from utils import preprocess_query, generate_embeddings, search_faiss_index
import numpy as np
import faiss

app = FastAPI()

index = faiss.read_index("faiss_ivf.index")
trial_ids = np.load("trial_ids.npy")

# Input model for user query
class Query(BaseModel):
    title: str
    primary_outcome: str
    secondary_outcome: str
    eligibility: str

@app.post("/search")
def search_trials(query: Query, k: int = 10):
    # Preprocess and embed the user query
    user_query = preprocess_query(query.title, query.primary_outcome, query.secondary_outcome, query.eligibility)
    query_embedding = generate_embeddings([user_query])
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    # Perform FAISS search
    results = search_faiss_index(query_embedding, index, trial_ids, k)
    return {"results": results}
