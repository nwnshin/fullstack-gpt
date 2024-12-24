# LLM이 파일 내용 관련 퀴즈 생성하고 정답 확인 - 정답에 대한 output 형태 parse
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.retrievers import WikipediaRetriever
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.callbacks import StreamingStdOutCallbackHandler # 모델의 실시간 답변을 출력해주는 콜백핸들러
from langchain.schema import BaseOutputParser
import json

st.set_page_config(page_title="QuizGPT", layout="centered")
st.title("Quiz GPT")

# output parser: string에서 json이나 ```를 삭제하고 파이썬 object로 만들어서 반환
# ```json 이 형식으로 prompt 전달하면 ai가 부가적인 말을 하지 않고 이대로만 답변해서 유용함
class JsonOutputParser(BaseOutputParser):
  def parse(self, text):
    text = text.replace("```","").replace("json","")
    return json.loads(text)

output_parser = JsonOutputParser()

# .cache_data()는 밑의 함수를 보고 해당 함수의 parameter를 hash한다. 
# hash라는 건 들어오는 데이터의 서명을 생성한다는 것이다. 
# 그래서 이 함수를 또 호출했을 때 parameter이 동일하면 서명이 동일하기에 함수를 run하지 않고 예전 값을 그대로 준다.
# 파일에 대해서는 서명 못만든다. 그래서 docs 대신 _docs하면 서명을 안만든다.
@st.cache_data(show_spinner="Loading...")
# 함수 split_file은 file이라는 변수를 받고, retriever을 반환함. 이번엔 벡터화해서 임베드 하지 않음.
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/quizfiles/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    # split
    splitted_docs = loader.load_and_split(text_splitter=splitter)
    return splitted_docs

# hash할 수 없는 매개변수가 있거나, streamlit이 데이터의 서명을 만들 수 없는 경우
# 다른 parameter를 하나더 넣어서, 이게 변경되면 streamlit이 함수를 재실행시키도록 함
# 다른 parameter안넣으면 docs가 변해도 맨날 똑같은 result만 나옴. 함수가 단 한번만 실행됨.
@st.cache_data(show_spinner="Making Quiz...")
def run_quiz_chain(_docs, topic):
    chain = {"document":question_chain} | format_chain | output_parser
    # chain.invoke(docs): array of questions 반환(파이썬에 array 타입 데이터인 object라서 활용 가능)
    return chain.invoke(docs)

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(topic):
  retriever = WikipediaRetriever(top_k_results=1, lang=lang)
  docs = retriever.get_relevant_documents(topic)
  return docs

def format_docs(retriever):
  return "\n\n".join(doc.page_content for doc in retriever)


# 사이드바 - apikey 입력, 난이도 조절, 파일 업로더or위키피디아 문서 검색
with st.sidebar:
  docs = None
  topic = None
  openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
  if openai_api_key:
    st.sidebar.write("Your API Key is set.")
  st.markdown("---")

  difficulty = st.selectbox(
    "Select Quiz Difficulty",
    options=["Easy", "Medium", "Hard"],
    index=0
  )

  choice = st.selectbox("Choose what content you want to use.", (
    "File", "Wikipedia Search"
  ))
  if choice == "File":
    file = st.file_uploader("Upload a file. (.docx, .txt, .pdf)", type=["pdf", "txt","docx"])
    if file:
      docs = split_file(file)
  else:
    topic = st.text_input("Search on Wikipidia")
    lang_choice = st.selectbox("Language",("EN","KO"))
    if topic:
      if lang_choice == "KO":
        lang = "ko"
      else:
        lang = "en"
      docs = wiki_search(topic)
        #wikidocs



llm = ChatOpenAI(streaming=True, temperature=0.3, model="gpt-4o-mini", callbacks=[StreamingStdOutCallbackHandler()])

# 프롬프트를 통해 LLM에 예시 제공
question_prompt = ChatPromptTemplate.from_messages([
  ("system","""
    You are a helpful assistant that is role playing as a teacher. 
    Based only on the following context make 10 questions to test the user's knowledge about the text.
    Each question should have 4 answers, three of them must be incorrect and one should be correct. 
    Use (o) to signal the correct answer. If document language is Korean, you should make it by Korean.

    Question examples:
      
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
          
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
          
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
          
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
          
    Your turn!

    document: {document}
    """)
])

question_chain = { "document":format_docs } | question_prompt | llm

# question chian으로 생성한 것을 json파일 형식format 으로s 가공 
# {{}} 이건 안해도 되는데 랭체인에 정확히 알려주려고 한 것. {document}랑 헷갈리지 말라고.
format_prompt = ChatPromptTemplate.from_messages([
  ("system", """
    You are a powerful formatting algorithm.
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
     
    Example Input:
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }},
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }},
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }},
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }},
                ]
            }}
        ]
     }}
    ```
    Your turn!
    Questions: {document}
  """
)])

format_chain = format_prompt | llm



# docs 변수가 없을 때 소개문 출력, 있을 때 해당 내용 출력
if not docs:
  st.markdown("""
    Welcome to Quiz GPT.

    Make a quiz from files you upload or from Wikipedia to test your knowledge and help you study. 
    Get started by uploading a file or searching on Wikipedia in the sidebar.
  """)
else:
  if not openai_api_key:
    st.error("Please enter your OpenAI API Key on the sidebar.")
  else:
  # 만약 유저가 file을 선택하면 topic은 없음. 그래서 topic if topic
    response = run_quiz_chain(docs, topic if topic else file.name)
    with st.form("questions_form"):
      for question in response["questions"]:
        st.write(question["question"])
        value = st.radio("Select an option.",
          [answer["answer"] for answer in question["answers"]],
          index=None)
        if {"answer":value, "correct": True} in question["answers"]:
          st.success("Correct!")
        elif value is not None:
          st.error("Wrong, try again.")
      button = st.form_submit_button() 
      # .form 위젯은 submit button이 필수적. 선택 후 submit버튼이 클릭되야 파일이 rerun하고 바뀐 부분이 인식되어 새 데이터를 갖게 됨.