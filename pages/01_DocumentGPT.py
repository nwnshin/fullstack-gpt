import streamlit as st
import time
import requests
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnablePassthrough

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

st.title("Document GPT")

llm = ChatOpenAI( temperature=0.3, streaming=True ) 

st.markdown("""
Welcome!

Use this chatbot to ask questions to an AI about your files!
You can upload your file on sidebar.
""")

# 함수 embed_file은 file이라는 변수를 받고, retriever을 반환함
@st.cache_data(show_spinner="Embedding...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    # split
    splitted_docs = loader.load_and_split(text_splitter=splitter)
    # 임베딩모델
    embeddings = OpenAIEmbeddings()
    # 캐싱
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    # 벡터화된 문서 저장
    vectorstore = FAISS.from_documents(splitted_docs, cached_embeddings)
    # retriever (검색기) 구현
    retriever = vectorstore.as_retriever()
    return retriever

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})

def format_docs(splitted_docs):
  return "\n\n".join(document.page_content for document in splitted_docs)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )

prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant. Answer questions using only the following context. If you don't know the answer just say you don't know, don't make it up:\n\n{document}"),
    MessagesPlaceholder(variable_name="history"),
    ("human","{question}")]
)

with st.sidebar:
  file = st.file_uploader("Upload a .txt, .pdf or .docx file", type=["pdf","txt","docx"])
  api_key = st.text_input("Enter your API Key:", type="password")
  if api_key:
    st.write("Your API Key is set.")
    headers = {"Authorization":f"Bearer {api_key}"}

if file: 
  retriever = embed_file(file)
  send_message("Ready! Ask me Anything.", "ai", save=False)
  paint_history()
  message = st.chat_input("Ask anything about your file...")
  if message:
    send_message(message, "human")
else:
  st.session_state["messages"] = []
