import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os

#path to the best model checkpoint (update as needed)
MODEL_PATH = '../models/flan_t5_lora_epoch3_COMBINATION1'
MODEL_NAME = 'google/flan-t5-base'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

model, tokenizer = load_model()

st.title('Mechanical Failure Solution Chatbot')
st.markdown('---')

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

user_input = st.text_input('Describe the failure or ask a question:', key='user_input')

if st.button('Send') and user_input.strip():
    #add user message to history
    st.session_state['chat_history'].append({'role': 'user', 'text': user_input})
    #prepare prompt
    prompt = f"Failure: {user_input} | Generate solution:"
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, padding='max_length', max_length=512).to(DEVICE)
    outputs = model.generate(**inputs, max_length=128)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #add model response to history
    st.session_state['chat_history'].append({'role': 'bot', 'text': response})
    st.session_state['user_input'] = ''
    st.experimental_rerun()

#chat bubbles
for msg in st.session_state['chat_history']:
    if msg['role'] == 'user':
        st.markdown(f"<div style='background-color:#DCF8C6; padding:10px; border-radius:10px; margin-bottom:5px; text-align:right;'><b>You:</b> {msg['text']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background-color:#F1F0F0; padding:10px; border-radius:10px; margin-bottom:5px; text-align:left;'><b>Bot:</b> {msg['text']}</div>", unsafe_allow_html=True) 