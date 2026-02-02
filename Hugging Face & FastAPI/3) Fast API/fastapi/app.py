'''
Deploying Hugging Face
• Hugging Face and Fast API
• Containerizing Hugging Face
• Running Fast API with Hugging Face
• Automating Packaging with Azure Container Registry
• Automating Packaging with Docker Hub
'''

from fastapi import FastAPI
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI()

# Load the Hugging Face pipeline for sentiment analysis
nlp_model = pipeline("sentiment-analysis")
#text = "I love ai "
#res = nlp_model(text)
#print(res)

text_generator = pipeline("text-generation", model="distilgpt2") #gpt

# Define a route for sentiment prediction
@app.get("/predict")
def predict(text: str):
    return nlp_model(text)

@app.get("/gen")
def gen(topic: str):
    generated_output = text_generator(topic, max_length=100, num_return_sequences=1)
    generated_text = generated_output[0]['generated_text']
    return generated_text