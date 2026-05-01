from langchain_core.prompts import ChatPromptTemplate

full_prompt = ChatPromptTemplate.from_messages(
    [
       ("system", "انت مساعد عربي ذكي يساعد المستخدم في الإجابة على الأسئلة. وحافظ على سياق المحادثة"),
        ("human", "{user_question}"),
        ("placeholder", "{History}")
    ]
    
)