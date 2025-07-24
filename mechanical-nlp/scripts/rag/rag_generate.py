import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from retrieve_similar_cases import retrieve_similar

MODEL_PATH = '../../models/flan_t5_lora_epoch3'
MODEL_NAME = 'google/flan-t5-base'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if __name__ == '__main__':
    query = input('Enter failure description: ')
    similar_cases = retrieve_similar(query, k=3)
    context = '\n'.join(similar_cases)
    prompt = f"Failure: {query} | Similar cases: {context} | Generate solution:"
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, padding='max_length', max_length=512).to(DEVICE)
    outputs = model.generate(**inputs, max_length=128)
    solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print('Generated Solution:')
    print(solution) 