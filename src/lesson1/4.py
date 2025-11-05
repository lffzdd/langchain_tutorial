from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.llm import deepseek,vllm

prompts=ChatPromptTemplate.from_messages([
    ("system", "你是阿强，一个乐于助人的助手。"),
    ("user", "{text}")
])

llm=vllm

chain=prompts | llm | StrOutputParser()

# chain.batch是重复invoke的简便方法
texts=[
    {'text':'你为什么喜欢萝莉'},
    {'text':'你为什么喜欢洗脚'}
]

responses=chain.batch(texts)
print(responses)
