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
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

st.title("Document GPT")

# .sidebar 메소드 실행
with st.sidebar:
    api_key = st.text_input("Enter your API Key:", type="password")
    if api_key:
        st.write("Your API Key is set.")
    else:
        st.error("Please enter your OpenAI API Key to proceed.")
    st.markdown("---")

    file = st.file_uploader("Upload a .txt, .pdf or .docx file", type=["pdf","txt","docx"])

class CallbackHandler(BaseCallbackHandler):
    message = ""
    # args = arguments , kwargs = keyword arguments
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
        st.write(">> llm started... <<")

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
        # st.sidebar.write("~") 과 동일
        with st.sidebar:
            st.write(">> llm finished. <<")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

llm = ChatOpenAI( temperature=0.3, streaming=True, callbacks=[CallbackHandler()], openai_api_key=api_key ) 

st.markdown("""
Welcome!

Use this chatbot to ask questions to an AI about your files!
You can upload your file on sidebar.
""")

# 함수가 처음 실행될 대 로딩표시 스피너 출력 
# 한번 실행 후 결과를 캐시에 저장하여 또 호출할 때 로딩없이 빠르게 반환함
@st.cache_data(show_spinner="Embedding...")
# 함수 embed_file은 file이라는 변수를 받고, retriever을 반환함
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
    embeddings = OpenAIEmbeddings(openai_openai_api_key=openai_api_key)
    # 캐싱
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    # 벡터화된 문서 저장
    vectorstore = FAISS.from_documents(splitted_docs, cached_embeddings)
    # retriever (검색기) 구현
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
        st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    # 채팅형태로 메세지를 화면에 출력함
    with st.chat_message(role):
        st.markdown(message)
    # save=True라면 메시지 내역을 session_state[messages]에 저장함(대화 기록 저장)
    if save:
        save_message(message, role)

def format_docs(retriever):
  return "\n\n".join(doc.page_content for doc in retriever)

def paint_history():
    # 저장된 메세지 내역을 화면에 출력함. 이미 저장된 내용이기 때문에 save=false로 또 저장되는 것 방지.
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )

# 대화 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful assistant. Answer questions using only the following context. If you don't know the answer just say you don't know, don't make it up:\n\n{document}"),
    ("human","{question}")]
)

def invoke():
    if not openai_api_key:
        return
    # 조건문: 파일 업로드 시 - retriever, 저장되지 않는 ai메세지, 대화 히스토리, input란 표시
    if file: 
        retrieved = embed_file(file) # 파일을 embed file 함수에 변수로 입력하고 출력받은 벡터화된 서류를 retrieved로 받음
        send_message("Ready! Ask me Anything.", "ai", save=False)
        paint_history()
        message = st.chat_input("Ask anything about your file...")
        # 조건문: 메세지 입력시 - send messages 함수 실행, retrieved된 벡터화된 내용에 메세지를 invoke하여 출력받은 list of 내용을 docs로 저장.
    if message:
        send_message(message, "human")
        chain = { "document": retrieved | RunnableLambda(format_docs), "question": RunnablePassthrough() } | prompt | llm
        with st.chat_message("ai"):
            response = chain.invoke(message)
        #send_message(response.content, "ai")
        #docs = retrieved.invoke(message)
        #join_docs = "\n\n".join(document.page_content for document in docs)
        #join_docs
        #prompt = template.format_messages(context=docs, question=message)
        #llm.predict_messages(prompt)
    # 조건문: 파일 없을 시 - session state 초기화
    else:
        st.session_state["messages"] = []
        return