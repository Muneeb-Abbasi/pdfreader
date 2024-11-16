import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
load_dotenv()

def handle_userinput(user_input):
    response = st.session_state.conversation({'question':user_input})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message("User"):
                st.write(message.content)  # User message
        else:
            with st.chat_message("Bot"):   
                st.write(message.content)  # Bot message

def get_pdf_text(pdf_docs): 
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorize_data(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key = os.getenv('OPENAI_API_KEY'))
    vectorstore = FAISS.from_texts(texts=chunks,embedding=embeddings)
    return vectorstore
    
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chain

def main():
    st.set_page_config(page_title="Chat With Your PDFS",page_icon=":books:")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    st.header("Chat With You PDFS :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your pdfs here and click on 'Process'",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorize_data(chunks)
                # st.write(vectorstore)
                st.session_state.conversation = get_conversation_chain(vectorstore)

                
if __name__ == '__main__':
    main()