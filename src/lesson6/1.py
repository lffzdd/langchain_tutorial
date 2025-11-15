from config.llm import deepseek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = deepseek

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的中文助手。"),
    ("placeholder", "{history}"),
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

# --- 手写“记忆” ---
history_messages = []

def talk(user_text: str) -> str:
    global history_messages

    # 1. 调用链时，把 history 消息显式传进去
    result = chain.invoke({
        "input": user_text,
        "history": history_messages,
    })

    # 2. 把本轮对话写回 history
    history_messages = history_messages + [
        HumanMessage(content=user_text),
        AIMessage(content=result),
    ]

    return result

print(talk("你好，我叫阿强。"))
print(talk("请记住我的名字。"))
print(talk("我是谁？"))
