import tempfile
import os
import shutil
import streamlit as st
from streamlit_chat import message
import requests


conversation_url = "http://localhost:8000/conversation"
vectorize_url = "http://localhost:8000/vectorize"
vectorstore_path = "./chroma_db"
if os.path.exists(vectorstore_path):
    shutil.rmtree(vectorstore_path, ignore_errors=True)

def conversational_chat(query):
    chat_history = st.session_state['history']
    request_json = {"question": query, "chat_history": chat_history}
    response = requests.post(conversation_url, json=request_json)
    response = response.json()
    st.session_state['history'].append((query, response["answer"]))
    return response["answer"]

def to_vectorstore(filepath):
    request_json = {"filepath": filepath}
    response = requests.post(vectorize_url, json=request_json)
    return



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
    
    if os.path.exists(vectorstore_path):
        shutil.rmtree(vectorstore_path, ignore_errors=True)
 
    st.write("Full path of the uploaded file:", temp_file_path)
    st.write("解析文章中")
    to_vectorstore(temp_file_path)
    st.write("解析完畢")

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
            
#streamlit run frontend.py






