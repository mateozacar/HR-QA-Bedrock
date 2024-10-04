import os
#Import the loader pdf
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain_aws import ChatBedrock

def hr_index():
    #Define the data source and load with PDF Loader
    data_load = PyPDFLoader("https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf")
    #split the text based on character, tokens etc.
    data_split=RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=100, chunk_overlap=10)
    #Embedding document
    data_embedding = BedrockEmbeddings(
        credentials_profile_name = "default",
        model_id = "amazon.titan-embed-text-v1")
    #Creating a vector DB store
    data_index = VectorstoreIndexCreator(
        text_splitter =  data_split,
        embedding = data_embedding,
        vectorstore_cls = FAISS)
    #Create index vector.
    db_index = data_index.from_loaders([data_load])    
    return db_index 

def hr_llm():
    llm=ChatBedrock(
       credentials_profile_name='default',
       model_id='anthropic.claude-v2:1',
       model_kwargs= {
           "max_tokens": 300,
           "temperature": 0.1,
           "top_p": 0.9} )
    return llm

def hr_rag_response(index,question):
    rag_llm = hr_llm()
    hr_rag_query = index.query(question=question,llm=rag_llm)
    return hr_rag_query

