import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAG_PATH = os.path.join(PROJECT_ROOT, 'scripts', 'rag')
sys.path.append(RAG_PATH)
from retrieve_similar_cases import retrieve_similar

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'flan_t5_lora_epoch3_COMBINATION1') 
MODEL_NAME = 'google/flan-t5-base'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

test_queries = [
    "sanitation not working",
    "conveyor belt stopped",
    "temperature sensor error",
    "hydraulic leak detected",
    "motor overheating",
    "unexpected shutdown",
    "pressure too low",
    "display screen blank"
]

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

for query in test_queries:
    print(f"\n=== QUERY: {query} ===")
    similar_cases = retrieve_similar(query, k=3)
    print("Retrieved similar cases:")
    for i, case in enumerate(similar_cases, 1):
        print(f"  {i}. {case}")
    context = '\n'.join(similar_cases)
    prompt = f"Failure: {query} | Similar cases: {context} | Generate a step-by-step solution for a technician to resolve this issue:"
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, padding='max_length', max_length=512).to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Model response: {response}") 