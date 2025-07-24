import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
INDEX_PATH = os.path.join(PROJECT_ROOT, 'rag', 'faiss_index.bin')
EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, 'rag', 'embeddings.npy')
INPUTS_PATH = os.path.join(PROJECT_ROOT, 'rag', 'inputs.npy')

model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(INDEX_PATH)
inputs = np.load(INPUTS_PATH, allow_pickle=True)
embeddings = np.load(EMBEDDINGS_PATH)

def retrieve_similar(query, k=3):
    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, k)
    return [inputs[i] for i in I[0]]

if __name__ == '__main__':
    query = input('Enter failure description: ')
    results = retrieve_similar(query, k=3)
    print('Top similar cases:')
    for r in results:
        print('-', r) 