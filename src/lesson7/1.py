from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 1. 初始化 LLM
planner_llm = ChatOpenAI(model="gpt-4o-mini")   # 负责“想计划”
executor_llm = ChatOpenAI(model="gpt-4o-mini")  # 负责“干活”
parser = StrOutputParser()


# 2. 构造 Planner 的 Prompt：只做一件事——把大任务拆成步骤
planner_prompt = ChatPromptTemplate.from_template("""
你是一个“任务规划助手”。
用户会给你一个整体任务，请你把它拆分成**有顺序的子任务列表**。

要求：
1. 子任务要尽量独立、明确，便于逐个执行。
2. 每个子任务前用 "Step N:" 标记（例如 Step 1: ...）。
3. 不要直接完成任务，只做“规划”。

用户任务：{user_task}
""")

planner_chain = planner_prompt | planner_llm | parser


# 3. 构造 Executor 的 Prompt：对每个子任务，认真完成
executor_prompt = ChatPromptTemplate.from_template("""
你是一个“任务执行助手”。

当前的**总目标**是：
{overall_goal}

现在需要你执行的**子任务**是：
{current_step}

请你只针对这个子任务给出详细的执行结果。
如果需要，可以用列表、小标题等结构化输出。
""")

executor_chain = executor_prompt | executor_llm | parser


def parse_steps(planner_output: str) -> List[str]:
    """
    非严格解析：把 LLM 输出里以 'Step ' 开头的行当成子任务。
    你后面也可以改成更严格的 JSON 输出。
    """
    steps = []
    for line in planner_output.splitlines():
        line = line.strip()
        if line.lower().startswith("step "):
            # e.g. "Step 1: 介绍 LangChain 是什么"
            # 去掉 "Step X:" 前缀
            parts = line.split(":", 1)
            if len(parts) == 2:
                steps.append(parts[1].strip())
            else:
                steps.append(line)
    # 如果解析不到，就把整体当成一个步骤
    if not steps and planner_output.strip():
        steps = [planner_output.strip()]
    return steps


def plan_and_execute(user_task: str) -> Dict[str, Any]:
    # 4. 调用 Planner，生成任务规划
    plan_text = planner_chain.invoke({"user_task": user_task})
    print("=== 规划结果 ===")
    print(plan_text)
    print("================")

    steps = parse_steps(plan_text)
    print("解析得到的步骤：", steps)

    # 5. 依次执行每个子任务
    results = []
    for i, step in enumerate(steps, start=1):
        print(f"\n--- 正在执行 Step {i}: {step} ---")
        result = executor_chain.invoke({
            "overall_goal": user_task,
            "current_step": step,
        })
        results.append({
            "step_index": i,
            "step_desc": step,
            "result": result,
        })

    # 6. 汇总
    return {
        "task": user_task,
        "plan_raw": plan_text,
        "steps": results,
    }


if __name__ == "__main__":
    user_task = "帮我整理一个关于 LangChain 核心模块（LLM、Prompt、Chain、Tool）的Markdown笔记大纲，并给出每个模块的简要解释。"
    final_output = plan_and_execute(user_task)

    print("\n\n=== 最终汇总（示例：简单合并） ===")
    for item in final_output["steps"]:
        print(f"\n## Step {item['step_index']}: {item['step_desc']}\n")
        print(item["result"])
