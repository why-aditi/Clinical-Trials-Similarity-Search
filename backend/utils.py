from transformers import AutoTokenizer, AutoModel
import torch

# Load the fine-tuned model
model_name = "fine_tuned_biobert_or_sentencebert"  # Replace with your model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def generate_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Pooling to get sentence embeddings
    return embeddings.numpy()

def search_faiss_index(query_embedding, index, trial_ids, k=10):
    # Query FAISS index
    D, I = index.search(query_embedding, k)
    # Retrieve trial IDs and distances
    results = [{"trial_id": trial_ids[i], "distance": float(D[0][j])} for j, i in enumerate(I[0])]
    return results

def preprocess_query(title: str, primary_outcome: str, secondary_outcome: str, eligibility: str) -> str:
    # Combine all query fields into a single text
    query = f"Title: {title}. Primary Outcome: {primary_outcome}. Secondary Outcome: {secondary_outcome}. Eligibility: {eligibility}."
    return query

