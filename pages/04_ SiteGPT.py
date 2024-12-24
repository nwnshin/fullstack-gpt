import streamlit as st
import requests
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import AsyncChromiumLoader

st.set_page_config(
  page_title="SiteGPT",
)
st.title("SiteGPT")
st.markdown("""
  #### Ask questions about the content of a website. 
  Start by writing the URL of the website on the sidebar. 
""")

llm = ChatOpenAI(temperature=0.5, streaming=True)

with st.sidebar:
  url = st.text_input("URL here", placeholder="https://example.com")

if url:
  # async chromium loader
  loader = AsyncChromiumLoader([url])
  docs = loader.load()
  docs