from fastapi import FastAPI
from chat import chat_system


fastAPI_APP = FastAPI()

#  uvicorn app:fastAPI_APP --reload --port 8005
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    user_question: str = Field(..., example="مرحبا")
    user_id: str = Field(..., example="d0117d2c-6b29-4d66-b1b9-b25ca4827696")




@fastAPI_APP.post("/chat_body")
def chat_endpoint(chat_request: ChatRequest):
    response = chat_system(chat_request.user_question, chat_request.user_id)
    return {"response": response}




@fastAPI_APP.post("/chat")
def chat_endpoint(user_question: str, user_id: str="d0117d2c-6b29-4d66-b1b9-b25ca4827696"):
    response = chat_system(user_question, user_id)
    return {"response": response}


@fastAPI_APP.get("/")
def read_root():
    return {"Hello": "Hello from FastAPI!"}

@fastAPI_APP.get("/health")
def read_health():
    return {"status": "healthy"}
