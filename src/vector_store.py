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

def adaptive_filter_gap(distances, indices, gap_threshold=0.1):
    scores = distances[0]
    ids = indices[0]

    filtered = [ids[0]]  # always keep the best result

    for i in range(1, len(scores)):
        gap = scores[i-1] - scores[i]

        # if big drop → stop
        if gap > gap_threshold:
            break

        filtered.append(ids[i])

    return filtered