from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from config.llm import deepseek
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

raw_docs = [
    (
        "店规",
        "足部按摩建议：先泡脚放松；轻重以舒适为准；如有疼痛或基础病史要提前说明。",
    ),
    ("天心区店铺", "天心区优选：足浴之家（晚9点前打烊），天心足疗馆，舒心足道。"),
]

# 1) 切分为小片段（chunk）——检索粒度
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
docs = []
for title, content in raw_docs:
    for chunk in splitter.split_text(content):
        docs.append(Document(page_content=chunk, metadata={"title": title}))

# 2) 向量化 + 存入向量库(本地Chroma)
embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    base_url="https://api.siliconflow.cn/v1",
    api_key="sk-ljuovztdpgqafjnufoxhqbuzfzyvctualkzjxekycvpywxpx",
)
vectordb = Chroma.from_documents(
    docs, embedding=embeddings, collection_name="foot_massage_store"
)

# 3) 封装成检索器
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# 4) 构造Prompt模板
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是严谨的中文助手。必须仅基于<检索材料>回答；若材料不足，请明确说明不足，避免编造。",
        ),
        (
            "user",
            "根据以下<检索材料>，回答用户的问题。\n<检索材料>：{context}\n用户问题：{question}\n\n 请给出简明可信的回答，并在末尾给出引用片段的标题列表。",
        ),
    ]
)

llm = deepseek
parser = StrOutputParser()


# 5) 检索+生成回答
def format_docs(docs: list[Document]):
    return "\n\n".join(
        [f"[{doc.metadata.get('title','无标题')}]\n{doc.page_content}" for doc in docs]
    )


rag_chain = (
    {
        "context": (lambda x: x["question"]) | retriever | format_docs,
        "question": lambda x: x["question"],
    }
    | prompt
    | llm
    | parser
)

# 6) 调用
print(rag_chain.invoke({"question": "在天心区有什么足部按摩店？给到注意事项。"}))
