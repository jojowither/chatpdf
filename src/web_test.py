import tempfile
import os
import shutil
from dotenv import load_dotenv, dotenv_values 
import streamlit as st
from streamlit_chat import message

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# 需處理不同文章時資料庫的更新

config = dotenv_values("../.env") 
os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY'] 
vectorstore_path = "./chroma_db"


# Create instance of OpenAI LLM
embeddings = OpenAIEmbeddings()


@st.cache_resource
def get_vectorstore(vectorstore_path, temp_file_path):
    if not os.path.exists(vectorstore_path):
        st.write("解析文章中")
        # Create and load PDF Loader
        loader = PyPDFLoader(temp_file_path)
        # Split pages from pdf 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
        pages = loader.load_and_split(text_splitter)

        # Load documents into vector database aka ChromaDB
        # vectorstore = Chroma.from_documents(pages, embeddings, collection_name='Pdf')
        vectorstore = Chroma.from_documents(pages, embeddings, persist_directory=vectorstore_path)
        st.write("解析完畢")
    else:
        vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)

    return vectorstore


@st.cache_resource
def get_chain():
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"), 
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        verbose=False
    )  
    return chain


@st.cache_data
def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]


# Set the title and subtitle of the app
st.title('🦜🔗 ChatPDF: 跟你的文件對話')
st.subheader('上傳PDF，問問題，然後從文件中獲得解答')

    
uploaded_file = st.file_uploader('', type=(['pdf',"tsv","csv","txt","tab","xlsx","xls"]))
while uploaded_file is None:
    x = 1
        
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())
 
    st.write("Full path of the uploaded file:", temp_file_path)
    vectorstore = get_vectorstore(vectorstore_path, temp_file_path)
    chain = get_chain()



if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["問我有關PDF內的問題!!! 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]
    
#container for the chat history
response_container = st.container()
#container for the user's text input
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        
        user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input', 
                                   label_visibility='collapsed',)
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        output = conversational_chat(user_input)
        
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
            
#streamlit run web_test.py
