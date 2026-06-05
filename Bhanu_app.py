import streamlit as st
import tempfile
import os
import time

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import (PyPDFLoader,TextLoader,CSVLoader,Docx2txtLoader)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
import trafilatura
from langchain_community.document_loaders import UnstructuredURLLoader, PlaywrightURLLoader

from duckduckgo_search import DDGS


# ------------------------------------------------
# ENV + LLM
# ------------------------------------------------

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key
)

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Multi-Document RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Bhanu Multi-Document RAG Assistant")

# ------------------------------------------------
# SESSION STATE INIT
# ------------------------------------------------

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.user_name:
    st.markdown("###👋 Welcome! Please enter your name to get started.")
    name_input = st.text_input("Your Name", placeholder="e.g. Bhanu")

    if st.button("Start Chat") and name_input.strip():
        st.session_state.user_name = name_input.strip()
        opening = (
        f"Hi! **{st.session_state.user_name}**... I'm an AI Assistant built by **Mr. Bhanu** 👋\n\n"
        "I'm here to help you with your questions. Here's what I can do:\n\n"
        "- 📄 **Upload documents** (PDF, DOCX, TXT, CSV) and ask questions from them\n"
        "- 🌐 **Paste a website URL** to chat with any webpage\n"
        "- 🔍 **Ask anything directly** — I'll search the web for you\n\n"
        "How can I help you today?")
        st.session_state.messages = [{"role":"assistant","content":opening}]
        st.rerun()
    st.stop()

if "langchain_history" not in st.session_state:
    st.session_state.langchain_history = []

# Display full chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ------------------------------------------------
# SIDEBAR — UPLOADS + URL
# ------------------------------------------------

uploaded_files = st.sidebar.file_uploader(
    "Upload Documents",
    type=["pdf", "docx", "txt", "csv"],
    accept_multiple_files=True
)

website_url = st.sidebar.text_input("Or paste a website URL")

if st.sidebar.button("Process Documents / URL"):
    # Store inputs in session state; actual build runs after helpers are defined below.
    st.session_state["trigger_build"] = True
    st.session_state["pending_files"] = uploaded_files
    st.session_state["pending_url"] = website_url

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.langchain_history = []
    st.rerun()

# ------------------------------------------------
# DOCUMENT LOADERS
# ------------------------------------------------

def load_documents(files):
    docs = []
    for uploaded_file in files:
        suffix = uploaded_file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        if suffix == "pdf":
            loader = PyPDFLoader(temp_path)
        elif suffix == "docx":
            loader = Docx2txtLoader(temp_path)
        elif suffix == "txt":
            loader = TextLoader(temp_path)
        elif suffix == "csv":
            loader = CSVLoader(temp_path)
        else:
            continue

        docs.extend(loader.load())
    return docs


def load_website(url: str):
    # Method 1: Trafilatura
    try:
        downloaded = trafilatura.fetch_url(url)
        extracted_text = trafilatura.extract(downloaded)
        if extracted_text:
            return [Document(page_content=extracted_text, metadata={"source": url})]
    except Exception as e:
        print(f"Trafilatura failed: {e}")

    # Method 2: Playwright
    try:
        loader = PlaywrightURLLoader(urls=[url])
        docs = loader.load()
        if docs and len(docs[0].page_content.strip()) > 100:
            return docs
    except Exception as e:
        print(f"Playwright failed: {e}")

    # Method 3: Unstructured
    try:
        loader = UnstructuredURLLoader(urls=[url])
        docs = loader.load()
        if docs and len(docs[0].page_content.strip()) > 100:
            return docs
    except Exception as e:
        print(f"Unstructured failed: {e}")

    raise Exception("Unable to extract content from the website.")


# ------------------------------------------------
# VECTOR STORE
# ------------------------------------------------

@st.cache_resource(show_spinner=False)
def create_vectorstore(all_docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    split_docs = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore


# ---- Trigger build after helpers are defined ----
if st.session_state.get("trigger_build"):
    st.session_state["trigger_build"] = False
    all_docs = []

    pending_files = st.session_state.get("pending_files") or []
    pending_url = st.session_state.get("pending_url", "")

    if pending_files:
        with st.spinner("Loading uploaded files..."):
            all_docs.extend(load_documents(pending_files))

    if pending_url:
        with st.spinner("Loading website content..."):
            try:
                all_docs.extend(load_website(pending_url))
                st.sidebar.success("Website loaded!")
            except Exception as e:
                st.sidebar.error(f"Website error: {e}")

    if all_docs:
        create_vectorstore.clear()
        with st.spinner("Building vector database..."):
            st.session_state.vectorstore = create_vectorstore(all_docs)

        # Reset chat on new document load
        st.session_state.messages = []
        st.session_state.langchain_history = []
        st.sidebar.success(f"Ready! Loaded {len(all_docs)} document chunk(s).")
    else:
        st.sidebar.warning("No documents or URL provided.")

# ------------------------------------------------
# PROMPTS
# ------------------------------------------------

# --- Greeting / chitchat detector ---
classify_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a classifier. Classify the user message into one of two categories:
CHITCHAT — greetings, thanks, small talk, personal questions, or anything not requiring information lookup.
QUESTION — any question or request that needs factual information, document lookup, or research.

Reply with ONLY one word: CHITCHAT or QUESTION. Nothing else.""",
        ),
        ("human", "{input}"),
    ]
)

# --- Chitchat / conversational response ---
chitchat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a friendly AI assistant. Respond naturally and conversationally. Keep it brief and warm.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# --- RAG (document) prompt ---
rag_prompt = ChatPromptTemplate.from_template(
    """You are a helpful AI assistant that answers questions based on provided document context.

If the answer exists in the context, answer using it.
If the answer is NOT in the context, respond with EXACTLY: NOT_FOUND_IN_DOCUMENTS

Context:
{context}

Question:
{input}"""
)

# --- Web search answer prompt (with history) ---
web_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant. Use the provided web search results to answer the user's question.
If the chat history is relevant, use it to provide a better answer.

Web Search Results:
{web_results}""",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# --- History-aware retriever prompt ---
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Given a chat history and the latest user question, rewrite the question as a standalone
question that can be understood without the chat history. Preserve all entities, names, and topics.
Do NOT answer. Only rewrite.""",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# ------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------

def classify_message(text: str) -> str:
    """Returns 'CHITCHAT' or 'QUESTION'."""
    chain = classify_prompt | llm
    result = chain.invoke({"input": text})
    label = result.content.strip().upper()
    return "CHITCHAT" if "CHITCHAT" in label else "QUESTION"


def get_chitchat_response(question: str, history: list) -> str:
    chain = chitchat_prompt | llm
    response = chain.invoke({"input": question, "chat_history": history})
    return response.content


def search_web(query: str, max_results: int = 5) -> str:
    result_texts = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        for r in results:
            result_texts.append(
                f"Title: {r.get('title', '')}\n"
                f"Content: {r.get('body', '')}\n"
                f"URL: {r.get('href', '')}"
            )
        return "\n\n".join(result_texts)
    except Exception as e:
        return f"Web search error: {e}"


def get_web_answer(question: str, history: list) -> str:
    """Search the web and answer, with chat history awareness."""
    web_results = search_web(question)
    chain = web_prompt | llm
    response = chain.invoke({
        "input": question,
        "web_results": web_results,
        "chat_history": history,
    })
    return response.content


def get_rag_answer(question: str, history: list):
    """Run RAG chain and return (answer, source_docs)."""
    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 8},
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    document_chain = create_stuff_documents_chain(llm, rag_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    response = retrieval_chain.invoke({"input": question, "chat_history": history})
    return response["answer"], response["context"]


# ------------------------------------------------
# CHAT INPUT & MAIN LOGIC
# ------------------------------------------------

user_question = st.chat_input("Ask anything — about your docs, the web, or just chat!")

if user_question:
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    answer = ""
    source_docs = []
    mode_label = ""

    start = time.perf_counter()

    # ── Step 1: Classify the message ──────────────────────────────────────
    msg_type = classify_message(user_question)

    if msg_type == "CHITCHAT":
        # Pure conversation — no search, no RAG
        answer = get_chitchat_response(user_question, st.session_state.langchain_history)
        mode_label = "💬 Conversational"

    elif "vectorstore" in st.session_state:
        # ── Step 2a: Try RAG first ─────────────────────────────────────────
        rag_answer, source_docs = get_rag_answer(
            user_question, st.session_state.langchain_history
        )

        if "NOT_FOUND_IN_DOCUMENTS" in rag_answer:
            # Not in docs → fall back to web search (with history)
            with st.spinner("Not found in documents. Searching the web..."):
                answer = get_web_answer(user_question, st.session_state.langchain_history)
            source_docs = []
            mode_label = "🌐 Web Search"
        else:
            answer = rag_answer
            mode_label = "📄 Document"

    else:
        # ── Step 2b: No docs loaded → web search (with history) ───────────
        with st.spinner("Searching the web..."):
            answer = get_web_answer(user_question, st.session_state.langchain_history)
        mode_label = "🌐 Web Search"

    elapsed = time.perf_counter() - start

    # Update LangChain history
    st.session_state.langchain_history.append(HumanMessage(content=user_question))
    st.session_state.langchain_history.append(AIMessage(content=answer))

    # Cap history to last 10 turns (20 messages) to avoid bloat
    if len(st.session_state.langchain_history) > 20:
        st.session_state.langchain_history = st.session_state.langchain_history[-20:]

    # Append and display assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)
        st.caption(f"{mode_label} · {elapsed:.2f}s")

        if source_docs:
            with st.expander("📎 Sources Used"):
                for i, doc in enumerate(source_docs, start=1):
                    st.markdown(f"**Source {i}**")
                    st.write(doc.metadata.get("source", "Unknown"))
                    st.write(doc.page_content[:500])
                    st.divider()