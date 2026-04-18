from urllib import response
from dotenv import load_dotenv
from groq import Groq 
import os 

load_dotenv()

def load_llm():

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return client

def generate_answer(client, context , query):
    prompt = f"""
You are an AI assistant.

Use the context below to answer the question.
You are allowed to infer meaning from the context.

If the answer is completely unrelated to the context, say "Not found in document".

Context:
{context}

Question:
{query}
"""
    response = client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature = 0.2

    )
    return response.choices[0].message.content
