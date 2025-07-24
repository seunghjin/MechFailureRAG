import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'train.csv')
INDEX_PATH = os.path.join(PROJECT_ROOT, 'rag', 'faiss_index.bin')
EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, 'rag', 'embeddings.npy')
INPUTS_PATH = os.path.join(PROJECT_ROOT, 'rag', 'inputs.npy')

model = SentenceTransformer(MODEL_NAME)
df = pd.read_csv(DATA_PATH)
inputs = df['input'].tolist()
embeddings = model.encode(inputs, show_progress_bar=True, convert_to_numpy=True)

np.save(EMBEDDINGS_PATH, embeddings)
np.save(INPUTS_PATH, np.array(inputs))

#FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, INDEX_PATH)
print(f'FAISS index built and saved to {INDEX_PATH}') 