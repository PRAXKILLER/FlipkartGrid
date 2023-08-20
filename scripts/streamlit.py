import sys
# sys.path.append("/home/ajay/Andy/shack-andy-bot/src")
sys.path.append("./src/pipeline")
from bot import *
import streamlit as st
from bot import *
from args import args
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import  SerpAPIWrapper
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter




chat_history = []
if __name__ == "__main__":

    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")

    st.title("💬 Eccomerce Assistant")
    if "messages" not in st.session_state:
        content = "Hi! What are you looking for today?"
        st.session_state["messages"] = [{"role": "assistant", "content": content}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        result = chatbot(
        {"question": prompt, "chat_history": chat_history}
            )
        msg = {"role": "assistant", "content": result["answer"]}
        print(msg)
        chat_history.append((result["question"], result["answer"]))
        st.session_state.messages.append(msg)
        st.chat_message("assistant").write(msg["content"])

