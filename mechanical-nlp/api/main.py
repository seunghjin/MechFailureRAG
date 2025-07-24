from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAG_PATH = os.path.join(PROJECT_ROOT, 'scripts', 'rag')
sys.path.append(RAG_PATH)
from retrieve_similar_cases import retrieve_similar

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'flan_t5_lora_epoch3_COMBINATION1')  # Update to your best model
MODEL_NAME = 'google/flan-t5-base'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

app = FastAPI()

# Enable CORS for all origins (including file://)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

# Load model and tokenizer once
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

@app.post('/chat')
async def chat_endpoint(req: ChatRequest):
    user_input = req.message
    similar_cases = retrieve_similar(user_input, k=3)
    context = '\n'.join(similar_cases)
    prompt = f"Failure: {user_input} | Similar cases: {context} | Generate a step-by-step solution for a technician to resolve this issue:"
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, padding='max_length', max_length=512).to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response} 