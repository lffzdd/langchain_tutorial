from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_classic.memory import ConversationBufferMemory
from config.llm import deepseek


@tool
def get_weather(city: str) -> str:
    """返回城市天气"""
    data = {"北京": "晴", "上海": "多云", "广州": "雷阵雨"}
    return data.get(city, "未知")


llm = deepseek
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

tools = [get_weather]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个友好的助手，可以调用工具查询天气。"),
        ("placeholder", "{chat_history}"),  # 注入记忆
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

# print(executor.invoke({"input": "你好，我在北京"}))
# print(executor.invoke({"input": "那广州的天气呢？"}))
# print(executor.invoke({"input": "我之前说我在哪个城市？"}))
