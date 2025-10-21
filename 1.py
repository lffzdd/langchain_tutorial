from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

prompt=ChatPromptTemplate.from_messages([
    ('system','你是阿强，一个乐于助人的助手。'),
    ('user','请把下面内容改写得更清晰但保留原意：{text}')
])

llm=ChatDeepSeek(
    model='deepseek-chat',
    temperature=0.7
)
