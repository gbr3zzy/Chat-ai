import os 
import streamlit as st
import openai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

def get_single_pdf_chunks(pdf, text_splitter):
    pdf_reader = PdfReader(pdf)
    pdf_chunks = []
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        page_chunks = text_splitter.split_text(page_text)
        pdf_chunks.extend(page_chunks)
    return pdf_chunks

def get_all_pdfs_chunks(pdf_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=500
    )

    all_chunks = []
    for pdf in pdf_docs:
        pdf_chunks = get_single_pdf_chunks(pdf, text_splitter)
        all_chunks.extend(pdf_chunks)
    return all_chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()  # Using OpenAI Embeddings instead of Google
    try:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.warning("Issue with reading the PDF/s. Your File might be scanned so there will be nothing in chunks for embeddings to work on")

def get_response(context, question, model_engine="gpt-4"):
    messages = [
        {"role": "system", "content": "You are a helpful and informative assistant that answers questions using text from the reference context."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ]

    try:
        response = openai.ChatCompletion.create(
            model=model_engine,
            messages=messages,
            max_tokens=8000,
            temperature=0.2,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response['choices'][0]['message']['content']

    except Exception as e:
        st.warning(e)

def working_process():

    vectorstore = st.session_state['vectorstore']

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a PDF Assistant. Ask me anything about your PDFs or Documents")
    ]
    
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

    user_query = st.chat_input("Enter Your Query....")
    if user_query is not None and user_query.strip() != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)
        
        with st.chat_message("AI"):
            try:
                relevant_content = vectorstore.similarity_search(user_query, k=10)
                result = get_response(relevant_content, user_query)
                st.markdown(result)
                st.session_state.chat_history.append(AIMessage(content=result))
            except Exception as e:
                st.warning(e)

def main():

    load_dotenv()

    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")
    st.header("Chat with Multiple PDFs :books:")

    openai.api_key = st.secrets["OPENAI_API_KEY"]

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and Click on 'Submit'", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing"):
                text_chunks = get_all_pdfs_chunks(pdf_docs)
                vectorstore = get_vector_store(text_chunks)
                st.session_state.vectorstore = vectorstore

    if st.session_state.vectorstore is not None:        
        working_process()

if __name__ == "__main__":
    main()
