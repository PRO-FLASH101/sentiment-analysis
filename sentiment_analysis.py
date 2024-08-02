from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Initialize the sentiment analysis pipeline with a specific model
pipe = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english")

# Create the FastAPI application
app = FastAPI() 

# Define the request model with Pydantic
class RequestModel(BaseModel):
    input: str

# Define the POST endpoint for sentiment analysis
@app.post("/sentiment")
def get_response(request: RequestModel):
    prompt = request.input
    response = pipe(prompt)
    label = response[0]["label"]
    score = response[0]["score"]
    return {"input": prompt, "label": label, "score": score}
    