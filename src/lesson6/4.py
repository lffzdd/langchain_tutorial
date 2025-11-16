from config.llm import deepseek, embeddings
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig


# ============= 1. 定义工具 (使用 @tool 装饰器,更简洁) =============
@tool
def vector_store_search(query: str) -> str:
    """执行向量数据库检索的工具函数。用于根据用户查询在向量数据库中检索相关信息。"""
    docs = [
        "passage:使用BGE并进行向量归一化有帮助",
        "passage:深度学习模型在NLP中很流行",
        "passage:向量数据库用于高效检索",
    ]

    # 自动计算文档嵌入并建立索引
    db = Chroma.from_texts(docs, embedding=embeddings)

    # 把向量库封装成检索器
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    # .invoke(query) 进行检索
    results = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in results])


@tool
def get_weather(location: str) -> str:
    """获取指定地点的天气信息。输入是地点名称，输出是该地点的天气情况。"""
    # 这里可以集成实际的天气API,这里只是示例
    data = {"北京": "晴", "上海": "多云", "广州": "雷阵雨"}
    return data.get(location, "未知")


# ============= 2. 创建带记忆和工具的 Agent =============
# InMemorySaver 自动管理对话历史 (记忆功能)
checkpointer = InMemorySaver()

agent = create_agent(
    model=deepseek,
    tools=[vector_store_search, get_weather],  # 传入工具列表
    system_prompt="你是一个乐于助人的助手，可以使用工具来获取信息，并给出有条理、简洁的中文答案。",
    checkpointer=checkpointer,  # 启用记忆功能
)


# ============= 3. 使用示例 =============
if __name__ == "__main__":
    # 配置 thread_id 用于区分不同对话会话
    config: RunnableConfig = {"configurable": {"thread_id": "user_001"}}

    print("=" * 60)
    print("第一轮对话: 工具调用 + 记忆")
    print("=" * 60)

    # 第一次调用: 使用工具查询天气
    response1 = agent.invoke(
        {"messages": [{"role": "user", "content": "北京的天气怎么样?"}]}, config
    )
    print("\n用户: 北京的天气怎么样?")
    print(f"助手: {response1['messages'][-1].content}\n")

    # 第二次调用: 使用工具查询向量数据库
    response2 = agent.invoke(
        {"messages": [{"role": "user", "content": "NLP中什么比较流行?"}]}, config
    )
    print("用户: NLP中什么比较流行?")
    print(f"助手: {response2['messages'][-1].content}\n")

    print("=" * 60)
    print("第二轮对话: 测试记忆功能")
    print("=" * 60)

    # 第三次调用: 测试记忆 - Agent 应该记得之前问的是北京的天气
    response3 = agent.invoke(
        {"messages": [{"role": "user", "content": "刚才我问的是哪个城市的天气?"}]},
        config,
    )
    print("\n用户: 刚才我问的是哪个城市的天气?")
    print(f"助手: {response3['messages'][-1].content}\n")

    print("=" * 60)
    print("查看完整的对话历史")
    print("=" * 60)

    # 查看完整的消息历史
    print(f"\n对话历史中共有 {len(response3['messages'])} 条消息:")
    for i, msg in enumerate(response3["messages"], 1):
        role = msg.__class__.__name__
        content_preview = (
            str(msg.content)[:80] + "..."
            if len(str(msg.content)) > 80
            else str(msg.content)
        )
        print(f"  {i}. [{role}] {content_preview}")
