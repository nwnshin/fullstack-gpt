import streamlit as st
import requests
import chardet
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import SitemapLoader # sitemaploader는 내부적으로 beautifulsoup을 사용해서 내용 편집을 함. 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_transformers import Html2TextTransformer # html을 받아서 text로 변환해줌
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough

llm = ChatOpenAI(temperature=0.5, streaming=True, openai_api_key=api_key)

html2text_transformer = Html2TextTransformer()

# 스크래핑한 데이터 편집하기
# beautiful soup object(html 덩어리)를 입력받음
# 이 function에서 반환한 값은 page_content로 데이터에 포함됨(?)
def parse_page(soup):
  header = soup.find("header")
  footer = soup.find("footer")
  if header:
    header.decompose()
  if footer:
    footer.decompose()
  return str(soup.get_text()).replace("/n"," ").replace("\xa0"," ") # header와 footer를 삭제한 데이터만 text로 반환

# 아래 함수가 한 번 실행된 후 같은 url로 또 호출되면 함수를 실행하지 않고, 이전의 캐시된(저장된) 반환값을 그대로 반환함.
@st.cache_data(show_spinner="Loading Website...")
def load_website(url):
  splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=200
  )
  # filter_urls는 list 혹은 정규식만 받음. 
  # 1. 데이터를 로드할 url의 list 2. 정규식(더 정밀한 필터링 가능, 추천)
  # r"^(?!.*\/blog\/).*" : /blog/를 포함하는 url은 전부 제외 
  # r"^(.*\/blog\/).*" : /blog/를 포함하는 url만 포함 filter_urls=[r"^(.*\/blog\/).*"], # 블로그 포스트만 데이터 스크랩하기
  loader = SitemapLoader(
    url, 
    parsing_function=parse_page
  )
  loader.request_per_second = 1 # 요청을 사이트에 보내는 속도 설정(1초에 1번). 너무 빠르면 차단 당한다.
  docs = loader.load_and_split(text_splitter=splitter) # 위에서  만든 splitter를 textsplitter로 사용
  vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
  return vector_store.as_retriever()

def encode_data(data):
  return

def get_answers(inputs):
  docs = inputs["docs"]
  question = inputs["question"]
  answers_chain = answers_prompt | llm

st.set_page_config(
  page_title="SiteGPT",
)
st.title("SiteGPT")
st.markdown("""
  #### Ask questions about the content of a website. 
  Start by writing the URL of the website on the sidebar. 
""")

with st.sidebar:
  api_key = st.text_input("Enter your OpenAI API Key:", type="password")
  if api_key:
    st.sidebar.write("Your API Key is set.")
  st.markdown("---")
  
  url = st.text_input("URL here", placeholder="https://example.com/sitemap.xml")

if url:
  # sitemaploader사용. url에 xml sitemap이 포함되어 잇는지 확인하는 로직
  if ".xml" not in url:
    with st.sidebar:
      st.error("URL has to be a Sitemap URL. Please try again.")
  else:
    retriever = load_website(url)
    
    # 체인 총 2개: 1개는 개별 docs에 대한 답변 생성과 채점, 나머지는 우리가 모든 답변을 받은 상태에 점수가 제일 높고 가장 최신인 답변으로 고르기
    # 체인1의 input: 검색기에 저장된 docs, 유저의 question
    chian = { "docs":retriever, "question": RunnablePassthrough() } | prompt | llm
    # 수집한 텍스트 편집하기
else:
  if not api_key:
    st.sidebar.error("Please enter your OpenAI API Key to proceed")
