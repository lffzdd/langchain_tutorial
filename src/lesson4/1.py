from config.llm import deepseek, vllm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool, BaseTool
from lesson4.tool import TOOLS, TOOL_MAP
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage,
    AIMessage,
)

from typing import List


llm = deepseek
llm = llm.bind_tools(TOOLS)


def run_agent(input: str, max_steps: int = 5):
    messages: List[BaseMessage] = [
        SystemMessage(content="你是阿强，一个乐于助人的洗脚大师。"),
        HumanMessage(content=input),
    ]

    for step in range(max_steps):
        ai: AIMessage = llm.invoke(messages)
        print(f"步骤 {step + 1} - AI 响应: {ai}")

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
                tool_result = f"未知工具: {function_name}"
            else:
                try:
                    tool_result = tool.invoke(tool_call.get("args"))
                except Exception as e:
                    tool_result = f"工具调用出错: {str(e)}"

            messages.append(
                ToolMessage(content=tool_result, tool_call_id=tool_call.get("id"))
            )

    # 如果达到最大步数仍未完成,返回最后一条消息
    return "未能生成回答"


if __name__ == "__main__":
    user_input = "你好，阿强！我最近脚很累，你能给我一些足部按摩的建议吗？另外，你能推荐一些天心区的足部按摩店铺吗？"
    response = run_agent(user_input)
    # print("最终回答：", response)
