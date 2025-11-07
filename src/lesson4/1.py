from config.llm import deepseek, vllm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
    AIMessage,
)
from typing import List, Dict


@tool
def get_foot_massage_tips() -> str:
    """获取足部按摩的技巧。"""
    return (
        "1. 挑选一个舒适的位置坐下。\n"
        "2. 呼叫几个服务员。\n"
        "3. 挑选一个中意的按摩师。\n"
        "4. 享受足部按摩的过程。\n"
    )


@tool
def get_foot_massage_shops(location: str) -> list[str]:
    """根据位置获取附近的足部按摩店铺推荐。"""
    # 长沙天心区，岳麓区，开福区
    shops = {
        "天心区": ["足浴之家", "天心足疗馆", "舒心足道"],
        "岳麓区": ["岳麓足疗中心", "养生堂足浴", "悦足坊"],
        "开福区": ["开福足疗馆", "康乐足道", "怡然足浴"],
    }
    return shops.get(location, ["抱歉，未找到该位置的足部按摩店铺推荐。"])


TOOLS: List[BaseTool] = [get_foot_massage_tips, get_foot_massage_shops]
TOOL_MAP: Dict[str, BaseTool] = {tool.name: tool for tool in TOOLS}

llm = deepseek
llm = llm.bind_tools(TOOLS)


def run_agent(input: str, max_steps: int = 5):
    messages: List[BaseMessage] = [
        SystemMessage(content="你是阿强，一个乐于助人的洗脚大师。"),
        HumanMessage(content=input),
    ]

    for step in range(max_steps):
        ai: AIMessage = llm.invoke(messages)

        # 没有工具调用，直接回答
        if not ai.tool_calls:
            messages.append(ai)
            return ai.content

        # 处理工具调用
        messages.append(ai)
        for tool_call in ai.tool_calls:
            function_name = tool_call.get("name")
            tool = TOOL_MAP.get(function_name)

            if tool is None:
                raise ValueError(f"未知工具: {function_name}")
            else:
                try:
                    tool_result = tool.invoke(tool_call.get("args"))
                except Exception as e:
                    tool_result = f"工具调用出错: {str(e)}"

            messages.append(
                ToolMessage(content=tool_result, tool_call_id=tool_call.get("id"))
            )

        return ai.content
