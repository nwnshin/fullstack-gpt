# LLM이 파일 내용 관련 퀴즈 생성하고 정답 확인 - 정답에 대한 output 형태 parse
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import BaseOutputParser

st.set_page_config(page_title="QuizGPT", layout="centered")
st.title("Quiz GPT")

with st.sidebar:
  api_key = st.text_input("Enter your OpenAI API Key:", type="password")
  if api_key:
    st.sidebar.write("Your API Key is set.")

  difficulty = st.selectbox(
    "Select Quiz Difficulty",
    options=["Easy", "Medium", "Hard"],
    index=0
  )

  choice = st.selectbox("Choose what you want to use.", (
    "file", "wikipedia article"
  ))
  if choice == "file":
    file = st.file_uploader("Upload a file. (.docx, .txt, .pdf)", type=["pdf", "txt","docx"])
  else:
    topic = st.text_input("Name of the article")