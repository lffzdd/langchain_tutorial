from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_core.runnables import RunnableSequence

messages = [
    ("system", "你是阿强，一个乐于助人的助手。"),
    ("user", "请把下面内容改写得更清晰但保留原意：{text}"),
]


llm = ChatDeepSeek(
    model="deepseek-chat",  # 使用 DeepSeek 的 chat 模型
    temperature=0.7,
    max_tokens=1024,
    timeout=30,
    max_retries=2,
    api_key="sk-4d60ba5196d14f939126d5e3b5f1647a",  # 如未设环境变量，可手动传
)

# ai_respond = llm.invoke(messages)
# print(ai_respond)

prompt = ChatPromptTemplate.from_messages(messages)
prompt = prompt.format_messages(text="萝莉住在萝莉塔里")
ai_respond = llm.invoke(prompt)
print(ai_respond.content)

prompt = ChatPromptTemplate.from_messages(messages)
chain=prompt | llm
ai_respond = chain.invoke({"text": "萝莉住在萝莉塔里"})
print(ai_respond.content)