# run:  uvicorn api:api --reload --host 127.0.0.1 --port 8000
# 
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app import chat, clear_session

api = FastAPI(title="HF LangChain Chat API", version="1.0")

class ChatRequest(BaseModel):
    session_id: str = Field(default="default")
    question: str = Field(..., min_length=1)

class ChatResponse(BaseModel):
    session_id: str
    answer: str

@api.get("/health")
def health():
    return {"status": "ok"}

@api.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    return ChatResponse(session_id=req.session_id, answer=chat(req.question, req.session_id))


from app import get_raw_history

@api.get("/history/{session_id}")
def history(session_id: str):
    return {"session_id": session_id, "history": get_raw_history(session_id)}

from pydantic import BaseModel, Field
from app import clear_session

class ClearRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Session identifier to clear")

@api.post("/clear")
def clear(req: ClearRequest):
    clear_session(req.session_id)
    return {"status": "cleared", "session_id": req.session_id}
