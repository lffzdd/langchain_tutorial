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
# 尝试禁用并行工具调用，看是否会有"思考过程"
llm = llm.bind_tools(TOOLS, parallel_tool_calls=False)


def run_agent(input: str, max_steps: int = 5):
    messages: List[BaseMessage] = [
        SystemMessage(content="你是阿强，一个乐于助人的洗脚大师。"),
        HumanMessage(content=input),
    ]

    for step in range(max_steps):
        ai: AIMessage = llm.invoke(messages)
        print(f"步骤 {step + 1} - AI 响应: {ai}")

        if not ai.tool_calls:
            messages.append(ai)
            return ai.content

        # 如果 AI 返回多个工具，只保留第一个（避免 tool_call_id 不匹配）
        if len(ai.tool_calls) > 1:
            ai = AIMessage(
                content=ai.content,
                tool_calls=[ai.tool_calls[0]],
                id=ai.id,
            )

        messages.append(ai)

        # 处理第一个工具（已确保 ai.tool_calls 只有一个元素）
        tool_call = ai.tool_calls[0]
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

    return "未能生成回答"


if __name__ == "__main__":
    user_input = "你好，阿强！我最近脚很累，你能给我一些足部按摩的建议吗？另外，你能推荐一些天心区的足部按摩店铺吗？"
    response = run_agent(user_input)
    # print("最终回答：", response)
