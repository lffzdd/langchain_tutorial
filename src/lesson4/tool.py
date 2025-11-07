from langchain_core.tools import tool, BaseTool

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
    # 天心区，岳麓区，开福区
    shops = {
        "天心区": ["足浴之家", "天心足疗馆", "舒心足道"],
        "岳麓区": ["岳麓足疗中心", "养生堂足浴", "悦足坊"],
        "开福区": ["开福足疗馆", "康乐足道", "怡然足浴"],
    }
    return shops.get(location, ["抱歉，未找到该位置的足部按摩店铺推荐。"])


TOOLS: List[BaseTool] = [get_foot_massage_tips, get_foot_massage_shops]
TOOL_MAP: Dict[str, BaseTool] = {tool.name: tool for tool in TOOLS}
