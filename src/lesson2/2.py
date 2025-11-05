from langchain_core.output_parsers import JsonOutputParser
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from config.llm import deepseek, vllm

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是严谨的摘要器，只输出指定结构，不要解释。"),
        (
            "user",
            "请从以下文本提取一个标题与3-5条要点：\n\n{text}\n\n"
            "严格遵循输出格式：\n{format_instructions}",
        ),
    ]
)

llm = vllm


class Summary(BaseModel):
    title: str = Field(..., description="文章标题")  # ...表示必填
    key_points: List[str] = Field(..., description="3~5条要点，每条一句话")



# json_parser = JsonOutputParser()
json_parser = JsonOutputParser(pydantic_object=Summary)

chain = prompt | llm | json_parser

doc = """
LangChain 让开发者把 LLM 应用拆成可复用部件：Prompt、模型、工具、检索与解析。
核心是 LCEL（LangChain Expression Language），像流水线一样组合步骤。
它也支持评测、缓存与并发，适合从 Demo 到生产的演进。
"""

result = chain.invoke(
    {"text": doc, "format_instructions": json_parser.get_format_instructions()}
)
print(result)
print(result.model_dump())
