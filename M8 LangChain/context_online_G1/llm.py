from langchain_openai import ChatOpenAI

from config import HF_TOKEN, MODEL,BASE_URL

llm_model = ChatOpenAI(model=MODEL
                 , api_key=HF_TOKEN
                 ,base_url=BASE_URL
                 ,temperature=0.2)

