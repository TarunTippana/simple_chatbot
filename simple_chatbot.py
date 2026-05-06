import streamlit as st
import dotenv
import langchain
import time
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
import os

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


st.set_page_config(page_title="Simple chatbot",page_icon="robot")
st.title("Chatbot with langchain and streamlit")




if "conv"  not in st.session_state:
    st.session_state["conv"]=[]
    st.session_state["memory"]=[]

    st.session_state["memory"].append(("system","You are a AI-ML Mentor and having experience of 25 years and you can give answer for my questions."))

for y in st.session_state["conv"]:
    with st.chat_message(y['role']):
        st.write(y["content"])



prompt = st.chat_input("Type your query:")
if prompt:
    st.session_state["conv"].append({"role":"user","content":prompt})
    st.session_state["memory"].append(("user",prompt))

    with st.chat_message("user"):
        st.write(prompt)       # to show the messages on the screen
    
    model=HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro")
    cmodel=ChatHuggingFace(llm=model)

    time.sleep(30) # for ratelimit error avoidance.
    response = cmodel.invoke(st.session_state["memory"])
    
    with st.chat_message("ai"):
        st.write(response.content)       # to show the messages on the screen


    st.session_state["conv"].append({"role":"ai","content":response.content})
    st.session_state["memory"].append(("user",response.content))

    
