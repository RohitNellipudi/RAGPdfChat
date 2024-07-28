import streamlit as st
from api_utils import readPdfIndex, getCoversationalChain, geneateResponse

import os
import getpass
import os

if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Provide your HUGGINGFACEHUB TOKEN")

# App title
st.set_page_config(page_title="🤗💬 KnowMeChat")

# function to clear the messages and start a new conversation
def clearMessages():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you with the pdf content?"}]

#created a sidebar to know abou the application and required libraries
with st.sidebar:
    st.title('🤗💬 RAG BAsed LLM chat application to answer queries from pdf')
    st.markdown('''
    ## About
    This app is an RAG based LLM powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [HuggingFace](https://huggingface.co/)
    - [LLM Model](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
    ''')    
    st.write("Application created by Rohit Nellipudi")
    st.header("Please tick the below checkbox to upload pdf")

# read more about Mistral 7B here - https://arxiv.org/pdf/2310.06825
model = "mistralai/Mistral-7B-Instruct-v0.2"

# Upload pdf UI elements
with st.sidebar:
    check = st.checkbox("upload pdf")
    if check:
        uploaded_file = st.file_uploader(label = "upload your dataset")
        if uploaded_file is not None:
            retriver = readPdfIndex(uploaded_file)
            st.write("successfully indexed")
            st.write(f"model used for chatbot: {model}")
            question_answerer = getCoversationalChain(model, retriver)


# UI elements to clear chat
with st.sidebar:
    st.button('Clear Chat', on_click = clearMessages)
    
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you with pdf content?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating LLM response
def generate_response(prompt_input):
    output = geneateResponse( question_answerer, prompt_input)
    return output

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)

