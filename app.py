from backend import *
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("openai_api_key")
os.environ['llm']=os.getenv("OpenAI_model")

llm = ChatOpenAI(temperature=0, model=os.getenv("OpenAI_model"))

directory_path = os.getenv("Folder_Path")

documents_all = load_documents(directory_path)
docs = split_documents(documents_all)

embeddings = OpenAIEmbeddings()
vectordb = FAISS.from_documents(docs, embeddings)

def handler(query):
    try:
        model = user_model(vectordb,llm)
        chat_history = []

        if query == '':
            return ''

        response = model.invoke({'question': query, 'chat_history': chat_history})
        chat_history.append(response['answer'])
        return response['answer']
    except Exception as e:
        return (f"An error occurred: {e}")
    

st.title("Chat with docs")

user_query = st.text_input("You:")

if user_query:
  response = handler(user_query)
  st.write("Assistant:", response)