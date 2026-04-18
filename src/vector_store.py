import faiss
import numpy as np

def create_faiss_index(embeddings):
    embeddings = np.array(embeddings).astype("float32")

    faiss.normalize_L2(embeddings) 

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index

def search(index, query_embedding, k=2):
    distances, indices = index.search(query_embedding, k)
    return distances, indices