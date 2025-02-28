"""Streamlit app"""
import os
from typing import Dict, List
import streamlit as st
import requests


HOST = "http://127.0.0.1:8501"  

st.title('MEDIA')



if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "What is your question ?"}]


for n, message in enumerate(st.session_state.messages):
    avatar = "🤖" if message["role"] == "assistant" else "🧑‍💻"
    st.chat_message(message["role"], avatar=avatar).write(
        message["content"])

if question := st.chat_input("What is your question ?"):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user", avatar="🧑‍💻").write(question)

    response = requests.post(
        f"{HOST}/answer",
        json={
            "question": question,
        },
        timeout=20
    )

    documents = requests.post(
        f"{HOST}/get_sources",
        json={
            "question": question,
        },
        timeout=20
    )

    if response.status_code == 200:
        answer = response.json()["message"]
        st.session_state.messages.append(
            {"role": "assistant", "content": answer})
        st.chat_message("user", avatar="🤖").write(answer)
    else:
        st.write("Error: Unable to get a response from the API")
        st.write(f"The error is: {response.text}")

