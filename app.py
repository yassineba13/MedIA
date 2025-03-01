import os
from typing import Dict, List
import streamlit as st
import requests

HOST = "https://yba-api-180940196448.europe-west1.run.app"
#HOST = "http://127.0.0.1:8501"

# Set page config
st.set_page_config(
    page_title="MedAI",
    page_icon="🩺",
    layout="centered"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        color: #0e4da4;
        text-align: center;
    }
    .subheader {
        color: #1e88e5;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        border-radius: 15px;
    }
    .disclaimer {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        font-size: 0.8rem;
        color: #555;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown("<h1 class='main-header'>🩺MedAI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Your personal healthcare assistant</p>", unsafe_allow_html=True)

# Medical disclaimer
with st.expander("Medical Disclaimer", expanded=False):
    st.markdown("""
    This AI assistant provides general medical information for educational purposes only.
    It is not a substitute for professional medical advice, diagnosis, or treatment.
    Always seek the advice of your physician or other qualified health provider with any
    questions you may have regarding a medical condition.
    """)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! I'm MedAI. How can I help with your medical questions today?"}
    ]

# Display chat messages
for n, message in enumerate(st.session_state.messages):
    avatar = "🩺" if message["role"] == "assistant" else "🧑"
    st.chat_message(message["role"], avatar=avatar).write(
        message["content"])

# Chat input
if question := st.chat_input("Ask your medical question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user", avatar="🧑").write(question)
    
    # Indicate loading state
    with st.spinner('Consulting medical knowledge...'):
        response = requests.post(
            f"{HOST}/answer",
            json={
                "question": question,
            },
            timeout=20
        )
    
    if response.status_code == 200:
        answer = response.json()["message"]
        st.session_state.messages.append(
            {"role": "assistant", "content": answer})
        st.chat_message("assistant", avatar="🩺").write(answer)
    else:
        st.error("Unable to connect to the medical database. Please try again later.")
        st.write(f"Error details: {response.text}")

# Footer with disclaimer
st.markdown("""
<div class="disclaimer">
    <strong>Important:</strong> The information provided by this assistant is not a substitute 
    for professional medical advice. If you're experiencing a medical emergency, please call 
    your local emergency number immediately.
</div>
""", unsafe_allow_html=True)