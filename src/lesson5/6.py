from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool, tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# --- Tool A: 天气（沿用你已有的） ---
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    data = {"北京": "晴，30°C", "上海": "多云，22°C", "广州": "雷阵雨，28°C"}
    return data.get(city, "未知城市")

# --- Tool B: 语义检索（把 retriever 暴露为工具） ---
class SearchInput(BaseModel):
    query: str = Field(..., description="简短中文查询语句")

def search_impl(query: str) -> str:
    docs = retriever.invoke(query)
    # 返回结构化文本，便于模型引用
    return "\n\n".join([f"【{d.metadata.get('title','无题')}】\n{d.page_content}" for d in docs])

search_tool = StructuredTool.from_function(
    func=search_impl,
    name="semantic_search",
    description="在本地知识库中进行语义检索，返回最相关片段。",
    args_schema=SearchInput,
)

# --- Agent：让模型自己决定是否需要检索/天气 ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
tools = [get_weather, search_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你可以调用工具。遇到知识型问题先尝试 semantic_search；遇到天气问题调用 get_weather。回答要引用片段标题并避免编造。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print(executor.invoke({"input": "给我推荐天心区的足部按摩店，并给到注意事项"})["output"])