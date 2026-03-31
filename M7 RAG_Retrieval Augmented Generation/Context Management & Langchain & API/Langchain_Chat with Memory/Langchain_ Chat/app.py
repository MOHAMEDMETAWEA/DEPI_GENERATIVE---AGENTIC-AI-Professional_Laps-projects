import os
from dotenv import load_dotenv
from typing import Dict

from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from prompts import chat_prompt

# Load environment variables from .env file
load_dotenv()

# Get credentials from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen3-Coder-Next:novita")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set it in your .env file or system environment.")

llm = ChatOpenAI(
    model=HF_MODEL,
    api_key=HF_TOKEN,
    base_url="https://router.huggingface.co/v1",
    temperature=0.2,
)

chat_chain = chat_prompt | llm  # Prompt → LLM

# dict: session_id -> {"lc_history": InMemoryChatMessageHistory, "raw_history": [{"role":..., "content":...}, ...]}
_STORE: Dict[str, Dict[str, any]] = {}

def _ensure_session(session_id: str):
    if session_id not in _STORE:
        _STORE[session_id] = {
            "lc_history": InMemoryChatMessageHistory(),
            "raw_history": []  # list of {"role": "human"/"ai", "content": "..."}
        }
    return _STORE[session_id]

def get_history(session_id: str):
    return _ensure_session(session_id)["lc_history"]

chat_with_memory = RunnableWithMessageHistory(
    chat_chain,
    get_history,
    input_messages_key="question",
    history_messages_key="history",
)

def chat(question: str, session_id: str = "default") -> str:
    question = (question or "").strip()
    if not question:
        return "اكتب رسالة أولاً."

    sess = _ensure_session(session_id)

    # سجّل رسالة المستخدم في raw_history
    sess["raw_history"].append({"role": "human", "content": question})

    # استدعاء LLM مع memory
    msg = chat_with_memory.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )
    answer = msg.content

    # سجّل رد المساعد في raw_history
    sess["raw_history"].append({"role": "ai", "content": answer})

    return answer

def get_raw_history(session_id: str = "default") -> list[dict]:
    return list(_ensure_session(session_id)["raw_history"])

def clear_session(session_id: str = "default") -> None:
    _STORE[session_id] = {
        "lc_history": InMemoryChatMessageHistory(),
        "raw_history": []
    }

if __name__ == "__main__":
    session_id = input("Session ID (مثلاً student1): ").strip() or "default"

    while True:
        q = input("\nسؤال (أو اكتب exit): ").strip()
        if q.lower() == "exit":
            break

        if q.lower() == "/history":
            print("\n=== RAW HISTORY (dict) ===")
            for i, item in enumerate(get_raw_history(session_id), 1):
                print(f"{i:02d}. [{item['role']}] {item['content']}")
            continue

        if q.lower() == "/clear":
            clear_session(session_id)
            print("تم مسح الذاكرة لهذه الجلسة.")
            continue

        print("\n---\n" + chat(q, session_id=session_id))