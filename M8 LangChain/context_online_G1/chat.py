from urllib import response

from llm import llm_model
from prompts import full_prompt

chat_chain= full_prompt | llm_model 

from db import get_history_from_postgres

from langchain_core.runnables.history import RunnableWithMessageHistory
# chat_chain= prompt | llm 
chat_with_history_postgres = RunnableWithMessageHistory(
    chat_chain,   
    get_history_from_postgres, 
    input_message_key="user_question",
    history_message_key="History"
)



def chat_system(user_question, user_id):
    response = chat_with_history_postgres.invoke(
        {"user_question": user_question},
        config={"configurable": {"session_id": user_id}}
    )
    return response