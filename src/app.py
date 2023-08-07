import os
import yaml
from dotenv import load_dotenv, dotenv_values 
from typing import List, Union, Tuple
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import tempfile

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate


config = dotenv_values("../.env") 
os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY'] 
vectorstore_path = "./chroma_db"
app = FastAPI()


def get_vectorstore_path(vectorstore_path, temp_file_path, embeddings):
    if not os.path.exists(vectorstore_path):
        # Create and load PDF Loader
        loader = PyPDFLoader(temp_file_path)
        # Split pages from pdf 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
        pages = loader.load_and_split(text_splitter)

        # Load documents into vector database aka ChromaDB
        # vectorstore = Chroma.from_documents(pages, embeddings, collection_name='Pdf')
        vectorstore = Chroma.from_documents(pages, embeddings, persist_directory=vectorstore_path)
    else:
        vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)

    return vectorstore



embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo')
chain = None


class FileModel(BaseModel):
    filepath: str


class ChatModel(BaseModel):
    question: str
    chat_history: Union[List[str], List[Tuple[str, str]]] = []


@app.post("/vectorize")
async def vectorize(data: FileModel):
    global chain
    vectorstore = get_vectorstore_path(vectorstore_path, data.filepath, embeddings)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        verbose=False
    )  
    return


@app.post("/conversation")
async def conversation(data: ChatModel):
    response = chain({"question": data.question, "chat_history": data.chat_history})
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)












