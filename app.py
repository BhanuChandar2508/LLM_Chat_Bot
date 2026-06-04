import streamlit as ST
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings


from dotenv import load_dotenv
load_dotenv()

## Load GROQ API KEy:
#os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
#groq_api_key = os.getenv("GROQ_API_KEY")
groq_api_key = ST.secrets["GROQ_API_KEY"]
llm = ChatGroq(groq_api_key=groq_api_key,model="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question:{input}

    """

    
)

## Create my Vectors:
@ST.cache_resource(show_spinner=False)
def _create_vector_embedding():
    ## Session state helps to remember vectorstore DB
    #if "vectors" not in ST.session_state:
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    loader=PyPDFDirectoryLoader("research") ## Data Ingestion Step
    docs=loader.load() ## Document Loading
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:])
    vector_store = FAISS.from_documents(final_documents,embeddings)

    return vector_store
    #else:
        #ST.write("Vector Embeddings were already ready. \n No Need to click Document Embeddings Button")

# Streamlit UI:
ST.set_page_config(
    page_title="RAG Document Q&A"
)
ST.title("RAG Document Q&A With GROQ")

# My modification:
with ST.spinner("Loading documents and creating vector store..."):
    vectors = _create_vector_embedding()
ST.success("Vector Database Ready Ask your query!...")

## User Query:

user_prompt = ST.text_input("Enter Your Query from the Doc:")
#if ST.button("Document Embeddings"):
#    create_vector_embedding()
#    ST.write("vector DataBase is Ready")
#else:
#    ST.write("Click Embeddings button, Before asking Query.")'''
import time

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriver=vectors.as_retriever()
    retriver_chain=create_retrieval_chain(retriver,document_chain)
    start_time = time.perf_counter()
    response=retriver_chain.invoke({"input":user_prompt})
    print(f"Response Time:{time.perf_counter()-start_time}")
    ST.write("### ANSWER:")
    ST.write(response["answer"])



    ## With A Streamlit expander:

    with ST.expander("Document Similarity Search"):
        for i,doc in enumerate(response['context'],start=1):
            ST.write(doc.page_content)
            ST.divider()



