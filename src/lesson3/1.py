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


llm = vllm
llm_with_tools = llm.bind_tools([get_weather])

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个乐于助人的助手，可以使用工具获取信息。"),
        ("user", "请告诉我{city}的天气情况，如果需要可以使用工具。"),
    ]
)

# chain = prompt | llm | StrOutputParser()
# result = chain.invoke({"city": "北京"})
# print("不使用工具的结果：", result)

chain_with_tools = prompt | llm_with_tools
result_with_tools = chain_with_tools.invoke({"city": "北京"})
print("\n使用工具的结果：", result_with_tools)
