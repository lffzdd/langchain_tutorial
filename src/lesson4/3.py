from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from config.llm import deepseek, vllm
from lesson4.tool import TOOLS


def build_agent(llm_model):
    """使用 LangGraph 预构建 ReAct Agent（最新推荐写法）。

    - llm_model: 任意支持工具调用的 ChatModel（如 ChatOpenAI、ChatDeepSeek 等）。
    - 返回：可直接 .invoke({"messages": [...]}) 的 agent 应用。
    - 提示：若使用本地 vLLM，需要在服务端开启自动工具选择（见下方注释）。
    """

    system_prompt = (
        "你是阿强，最喜欢洗脚。你可以调用工具来获取信息，并给出有条理、简洁的中文答案。"
    )

    # Python 端 create_react_agent 支持传入 prompt（或 state_modifier）作为系统指令
    agent_app = create_agent(llm_model, TOOLS, system_prompt=system_prompt)
    return agent_app


def demo_run(use_vllm: bool = False) -> None:
    """演示：用最新的 LangGraph 预构建 Agent 完成一次工具调用。"""

    # 如使用 vLLM，请确保：
    #   --enable-auto-tool-choice  与  --tool-call-parser <适配模型>
    # 例如 Qwen 系列：--tool-call-parser qwen2
    llm_model = vllm if use_vllm else deepseek

    agent = build_agent(llm_model)

    user_question = (
        "你好，阿强！我最近脚很累，你能给我一些足部按摩的建议吗？"
        "另外，你能推荐一些天心区的足部按摩店铺吗？"
    )

    messages = [HumanMessage(content=user_question)]
    result = agent.invoke({"messages": messages})

    print("\n===== Agent 对话消息 =====")
    for m in result["messages"]:
        try:
            # m.pretty_print()
            print(m)
            print('\n---\n')
        except Exception:
            # 某些消息类型可能无 pretty_print
            print(f"{m.type}: {getattr(m, 'content', m)}")


if __name__ == "__main__":
    demo_run(use_vllm=False)
