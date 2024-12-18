import streamlit as st
from datetime import datetime

st.set_page_config(
  page_title="fullstackGPT HOME",
  page_icon="💖"
)

st.title("Main")

today = datetime.today().strftime("%H:%M:%S")
today

st.write("""
  # Hello!

  Welcome to my Fullstack GPT Portfolio!

  Here are the apps I made:

  - [ ] [DocumentGPT](/DocumentGPT)
  - [ ] [PrivateGPT](/PrivateGPT)
  - [ ] [QuizGPT](/QuizGPT)
""")