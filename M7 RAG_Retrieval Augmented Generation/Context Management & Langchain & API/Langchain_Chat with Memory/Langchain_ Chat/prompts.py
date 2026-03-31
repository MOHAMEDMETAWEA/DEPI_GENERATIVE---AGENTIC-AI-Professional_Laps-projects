from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "أنت مساعد عربي واضح ومفيد. حافظ على سياق المحادثة."),
    ("placeholder", "{history}"),
    ("human", "{question}")
])