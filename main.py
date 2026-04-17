from src.embeddings import load_model, get_embeddings
from sklearn.metrics.pairwise import cosine_similarity

sentences=["AI is  transforming the world ","Natural Language Processing is a subfield of Machine Learning" , "I love trying new foods"]

model = load_model()
embeddings = get_embeddings(model, sentences)

similarity = cosine_similarity([embeddings[0]], embeddings)
print(similarity)