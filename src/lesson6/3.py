from config.llm import deepseek
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

# 1. 定义模型与 prompt
llm = deepseek
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个乐于助人的助手。"),
        ("user", "{input}"),
        ("placeholder", "{history}"),  # 占位符,用于注入历史对话
        # ("assistant", "{history}"),  # 占位符,用于注入历史对话
    ]
)

chain = prompt | llm | StrOutputParser()

store = {}


# 2. 定义“记忆存储”,InMemoryChatMessageHistory 保存在内存中,适合测试和小规模使用
def get_history(user_id: str) -> InMemoryChatMessageHistory:
    if user_id not in store:
        store[user_id] = InMemoryChatMessageHistory()
    return store[user_id]

# 3. 定义带有记忆的 Runnable

chain_with_memory = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 4.调用,同一个 thread_id 会记住之前的对话
config=RunnableConfig(configurable={'session_id':'conversation_1'})
# config={'configurable':{'session_id':'conversation_1'}}

response1 = chain_with_memory.invoke({"input": "你好,我叫阿强"}, config=config)
print("回复1:", response1)
response2 = chain_with_memory.invoke({"input": "请记住我的名字?"}, config=config)
print("回复2:", response2)
response3 = chain_with_memory.invoke({"input": "我是谁?"}, config=config)
print("回复3:", response3)