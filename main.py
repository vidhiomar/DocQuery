from src.embeddings import load_model, get_embeddings
import numpy as np
from src.vector_store import search , create_faiss_index , adaptive_filter_gap
from src.llm import load_llm, generate_answer
import faiss



sentences=["AI is  transforming the world ","Natural Language Processing is a subfield of Machine Learning" , "I love trying new foods"]

model = load_model()
client = load_llm()

embeddings = get_embeddings(model, sentences)
index=create_faiss_index(embeddings)

query = ["What is AI?"]
query_embedding = model.encode(query)

query_embedding = np.array(model.encode(query).astype("float32"))
faiss.normalize_L2(query_embedding)

distances, indices = search(index, query_embedding, k=5)
filtered_indices = adaptive_filter_gap(distances , indices)

if not filtered_indices:
    filtered_indices = indices[0][:1]

context = "\n".join([sentences[i] for i in filtered_indices])

answer = generate_answer(client , context, query )

print("\nContext used:\n", context )
print("\nFinal Answer:\n", answer)


# print("Results:")
# for i in indices[0]:
#     print(sentences[i])


# similarity = cosine_similarity([embeddings[0]], embeddings)
# print(similarity)