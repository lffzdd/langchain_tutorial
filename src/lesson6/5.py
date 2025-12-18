# -*- coding: utf-8 -*-
from typing import List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from config.llm import deepseek
from config.tool import get_weather, vector_store_search

# --- 定义工具 ---
tools = [get_weather, vector_store_search]

# --- 模型与提示模板 ---
llm = deepseek  # 替换为 deepseek 如有需要


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个乐于助人的中文助手，可以使用工具查询信息。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# --- 创建 Agent ---
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 配置记忆支持 ---
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


agent_with_memory = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# --- 调用演示 ---
config = {"configurable": {"session_id": "user_001"}}

res1 = agent_with_memory.invoke({"input": "请告诉我北京的天气。"}, config=config)
print("回答1:", res1)

res2 = agent_with_memory.invoke(
    {"input": "刚才你说北京天气，现在广州呢？"}, config=config
)
print("回答2:", res2)

res3 = agent_with_memory.invoke({"input": "我之前在哪个城市？"}, config=config)
print("回答3:", res3)
