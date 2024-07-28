from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

#API to read pdf conetent chunk and store in a vector database
def readPdfIndex(uploaded_content):
    # pdf reader to read pdf content
    pdf_reader = PdfReader(uploaded_content) 
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Langchain's Recursive character text splitter for splitting into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
    
    chunks = text_splitter.split_text(text=text)

    #FAISS(Facebook AI Similarity Search) vector db with Hugging Face embeddings to index the document, offers faster retriveal vs chroma db
    db = FAISS.from_texts(chunks, embedding=HuggingFaceEmbeddings())
    return db

def getCoversationalChain(model_id, docsearch):

    # modifu db to retriver 
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":4})

    # Hugging Face endpoint of Mistral7B model, temperature is set low as we need information from the pdf rather than generating new content
    llm=HuggingFaceEndpoint(repo_id=model_id, temperature=0.1, max_length=512)

    # Prompt template generation 
    prompt_template = """Text: {context}
    Question: {question}
    you are a chatbot designed to assist the users.
    Answer only the questions based on the text provided. If the text doesn't contain the answer,
    reply that the answer is not available.
    keep the answers precise to the question"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = { "prompt" : PROMPT }

    # conversation buffer memory to keep track of the conversation
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    #create langchain conversation chain for querying LLM
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever= retriever,
        memory = memory,
        combine_docs_chain_kwargs=chain_type_kwargs
    )
    return conversation_chain

# API to generate response for the query usign chain
def geneateResponse(chain, message):
    response  = chain({"question": message})
    return(response.get("answer"))