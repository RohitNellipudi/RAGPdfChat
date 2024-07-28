# RAGBasedLLM

This a RAG based LLM Chatbot - created by **Rohit Nellipudi**

- you can upload PDF to the app and ask questions related to the applications and maintain conversations 

- This app is an RAG based LLM powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [HuggingFace](https://huggingface.co/)
    - [Mistral 7B LLM Model](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)


**More details about Application**
- Build the applicaation using langchain
- Used FIASS DB for vector database 
- Hugging Face embeddings for creating embeddings and stroing in DB
- Used Mistral7B Hugging face model as LLM for converations
- Uses Streamlit for the UI elements of the application



**Requirements to run the applications** 
- Refer to requirements.txt for required installations
- Need Hugging Face API key to run the application 
- Run "streamlit run app.py" in the console to execute APP  


## Running the application 
```
streamlit run app.py
```
- Application will promt for Hugging Face API key
- Proivde the key in the console 
- Application will open in the browser
- Upload your PDF
- Ask questions related to your PDF

