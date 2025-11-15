from config.llm import deepseek
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig

# 创建带有记忆的 agent (替代旧的 ConversationChain + ConversationBufferMemory)
checkpointer = InMemorySaver()
agent = create_agent(
    model=deepseek,
    tools=[],  # 如果不需要工具,传入空列表
    system_prompt="你是一个乐于助人的助手。",  # 相当于之前的 system message
    checkpointer=checkpointer,  # 自动管理对话历史
)

# 使用示例
if __name__ == "__main__":
    config: RunnableConfig = {
        "configurable": {"thread_id": "conversation_1"}
    }  # thread_id 用于区分不同对话

    # 第一轮对话
    response1 = agent.invoke(
        {"messages": [{"role": "user", "content": "你好,我叫小明"}]}, config
    )
    print("回复1:", response1["messages"][-1].content)

    # 第二轮对话 - agent 会记住之前的对话
    response2 = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫什么名字?"}]}, config
    )
    print("回复2:", response2["messages"][-1].content)
