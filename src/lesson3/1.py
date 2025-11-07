from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.llm import deepseek, vllm


@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    # 这里可以调用实际的天气API，这里用静态数据模拟
    weather_data = {
        "北京": "晴，30°C",
        "上海": "多云，22°C",
        "广州": "雷阵雨，28°C",
    }
    return weather_data.get(city, "未知城市")


print(get_weather.name)
print(get_weather.description)
print(get_weather.args)

llm = deepseek
llm_with_tools = llm.bind_tools([get_weather])

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个乐于助人的助手，可以使用工具获取信息。"),
        ("user", "请告诉我{city}的天气情况，如果需要可以使用工具。"),
    ]
)

chain_with_tools = prompt | llm_with_tools
first_response = chain_with_tools.invoke({"city": "北京"})
print(first_response)

from langchain_core.messages import AIMessage,ToolMessage

# 如果tool_calls包含get_weather的调用
for tool_call in first_response.tool_calls:
    if tool_call.get('name') == get_weather.name:
        print(tool_call)

        tool_result = get_weather.invoke(tool_call.get('args'))
        print(f"Tool result: {tool_result}")

        tool_message=ToolMessage(
            content=tool_result,
            tool_call_id=tool_call.get('id')
        )

        print(tool_message)
        print("-----继续对话-----")
        print([first_response,tool_message])
        second_response = llm_with_tools.invoke([first_response,tool_message])
        print(second_response)