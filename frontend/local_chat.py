import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAG_PATH = os.path.join(PROJECT_ROOT, 'scripts', 'rag')
sys.path.append(RAG_PATH)
from retrieve_similar_cases import retrieve_similar

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'flan_t5_lora_epoch3_COMBINATION1')  # Update to your best model
MODEL_NAME = 'google/flan-t5-base'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print('Mechanical Failure Solution RAG Chat (type "exit" to quit)')

while True:
    user_input = input('You: ')
    if user_input.strip().lower() == 'exit':
        print('Exiting chat.')
        break
    # RAG: retrieve similar cases
    similar_cases = retrieve_similar(user_input, k=3)
    context = '\n'.join(similar_cases)
    prompt = f"Failure: {user_input} | Similar cases: {context} | Generate solution:"
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, padding='max_length', max_length=512).to(DEVICE)
    outputs = model.generate(**inputs, max_length=128)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f'Bot: {response}\n') 