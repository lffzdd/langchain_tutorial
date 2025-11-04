# 方式一
# LangChain为各llm提供了统一的接口：ChatModel
from langchain_deepseek.chat_models import ChatDeepSeek

llm = ChatDeepSeek(model="deepseek-chat", api_key="sk-4d60ba5196d14f939126d5e3b5f1647a")

response = llm.invoke("给我讲个关于量子纠缠的笑话")
print(response.content)

# 方式二
# 通过链式结构
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("给我讲个关于{topic}的笑话")

model = ChatDeepSeek(
    model="deepseek-chat", api_key="sk-4d60ba5196d14f939126d5e3b5f1647a"
)
chain = prompt | model

response = chain.invoke({"topic": "量子纠缠"})
print(response.content)
