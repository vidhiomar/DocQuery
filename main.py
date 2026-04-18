from src.embeddings import load_model, get_embeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.vector_store import search , create_faiss_index
from src.llm import load_llm, generate_answer

sentences=["AI is  transforming the world ","Natural Language Processing is a subfield of Machine Learning" , "I love trying new foods"]

model = load_model()
client = load_llm()

embeddings = get_embeddings(model, sentences)
index=create_faiss_index(embeddings)

query = ["What is AI?"]
query_embedding = model.encode(query)

distances, indices =search(index, np.array(query_embedding), k=1)

context =""
for i in indices[0]:
    context += sentences[i] + "\n"

answer = generate_answer(client , context, query )

print("\nContext used:\n", context )
print("\nFinal Answer:\n", answer)


# print("Results:")
# for i in indices[0]:
#     print(sentences[i])


# similarity = cosine_similarity([embeddings[0]], embeddings)
# print(similarity)