import os
import subprocess
import sys
import time
import webbrowser

# Paths
DATA_PATH = os.path.join('data', 'train.csv')
FAISS_PATH = os.path.join('rag', 'faiss_index.bin')
API_PATH = os.path.join('api', 'main.py')
CHAT_HTML_PATH = os.path.join('frontend', 'chat.html')

# 1. Check if data/train.csv exists
if not os.path.exists(DATA_PATH):
    print('Preparing data...')
    subprocess.run([sys.executable, os.path.join('scripts', 'data_preprocessing.py')])
    subprocess.run([sys.executable, os.path.join('scripts', 'split_data.py')])
else:
    print('Data already prepared.')

# 2. Check if rag/faiss_index.bin exists
if not os.path.exists(FAISS_PATH):
    print('Building FAISS index...')
    subprocess.run([sys.executable, os.path.join('scripts', 'rag', 'build_faiss_index.py')])
else:
    print('FAISS index already exists.')

# 3. Start FastAPI backend in a subprocess
print('Starting backend API...')
api_proc = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'api.main:app', '--reload'])

# Wait for backend to start
time.sleep(5)

# 4. Open chat.html in the default browser
chat_url = 'file://' + os.path.abspath(CHAT_HTML_PATH)
print(f'Opening chat UI at {chat_url}')
webbrowser.open(chat_url)

print('You can now chat in your browser. Press Ctrl+C here to stop the backend.')
try:
    api_proc.wait()
finally:
    print('Shutting down backend API...')
    api_proc.terminate() 