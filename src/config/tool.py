from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from config.llm import embeddings

# 预先构建好向量数据库和检索器
docs = [
    "passage:使用BGE并进行向量归一化有帮助",
    "passage:深度学习模型在NLP中很流行",
    "passage:向量数据库用于高效检索",
]
db = Chroma.from_texts(docs, embedding=embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})


@tool(
    "vector_store_search",
    description="执行向量数据库检索的工具函数。用于根据用户查询在向量数据库中检索相关信息。",
)
def vector_store_search(query: str) -> str:
    """执行向量数据库检索的工具函数。用于根据用户查询在向量数据库中检索相关信息。"""

    # .invoke(query) 进行检索
    results = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in results])


@tool(
    "get_weather",
    description="获取指定地点的天气信息。输入是地点名称，输出是该地点的天气情况。",
)
def get_weather(location: str) -> str:
    """获取指定地点的天气信息。输入是地点名称，输出是该地点的天气情况。"""
    data = {"北京": "晴", "上海": "多云", "广州": "雷阵雨"}
    return data.get(location, "未知")
