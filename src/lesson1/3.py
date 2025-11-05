"""
链式写作示例：根据不同风格要求，重写一段文字。
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import Literal

from config.llm import deepseek, vllm

style_map={
    "严谨": "使用正式、客观、学术化的语气；避免夸张修辞。",
    "科普": "面向非专业读者，举例贴近生活，避免术语或配合解释。",
    "广告": "突出卖点，强调利益点和行动号召，但不夸大其词。"
}

def make_writer(style:Literal["严谨", "科普", "广告"]="科普",max_tokens=100):
    sys=f'你是一名专业的写手，擅长{style_map[style]}，请用不超过{max_tokens}个字来写一篇{style}的文章。'
    prompt=ChatPromptTemplate.from_messages([
        ("system", sys),
        ("user", "{text}")
    ])
    
    llm=deepseek

    llm=vllm
    
    parser=StrOutputParser()

    return prompt | llm | parser
    
rewrite_chain=make_writer(style="严谨",max_tokens=100)

result=rewrite_chain.invoke({'text':'量子计算利用量子叠加与纠缠实现并行计算，但目前物理实现仍面临纠错与扩展性挑战。'})
print(result)
