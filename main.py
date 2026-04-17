from src.embeddings import load_model, get_embeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.vector_store import search , create_faiss_index

sentences=["AI is  transforming the world ","Natural Language Processing is a subfield of Machine Learning" , "I love trying new foods"]

model = load_model()

embeddings = get_embeddings(model, sentences)
index=create_faiss_index(embeddings)

query = ["food"]
query_embedding = model.encode(query)

distances, indices =search(index, np.array(query_embedding), k=2)
print("Results:")
for i in indices[0]:
    print(sentences[i])


# similarity = cosine_similarity([embeddings[0]], embeddings)
# print(similarity)