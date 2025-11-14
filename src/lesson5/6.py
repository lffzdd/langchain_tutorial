from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

raw_docs = [
    (
        "店规",
        "足部按摩建议：先泡脚放松；轻重以舒适为准；如有疼痛或基础病史要提前说明。",
    ),
    ("天心区店铺", "天心区优选：足浴之家（晚9点前打烊），天心足疗馆，舒心足道。"),
]

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
docs = []
for title, content in raw_docs:
    for chunk in splitter.split_text(content):
        docs.append(Document(page_content=chunk, metadata={"title": title}))

vector_db = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(
        model="BAAI/bge-m3",
        base_url="https://api.siliconflow.cn/v1",
        api_key="sk-ljuovztdpgqafjnufoxhqbuzfzyvctualkzjxekycvpywxpx",
    ),
    collection_name="foot_massage_store",
)

retriever=vector_db.as_retriever(search_kwargs={"k": 3})
