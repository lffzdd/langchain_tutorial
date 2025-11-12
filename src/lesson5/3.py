from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
import os

os.environ["OPENAI_API_KEY"] = "sk-ljuovztdpgqafjnufoxhqbuzfzyvctualkzjxekycvpywxpx"
embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3", base_url="https://api.siliconflow.cn/v1"
)

docs = [
    "passage:使用BGE并进行向量归一化有帮助",
    "passage:深度学习模型在NLP中很流行",
    "passage:向量数据库用于高效检索",
]
query = "query:NLP中什么比较普遍"

# 自动计算文档嵌入并建立索引
db=Chroma.from_texts(docs,embedding=embeddings)

# 把向量库封装成检索器
retriever=db.as_retriever(search_type="similarity",search_kwargs={"k": 2})

# .invoke(query) 进行检索
# 1.把query转换成向量
# 2.计算query向量和文档向量的相似度
# 3.排序取前k个最相似的文档
# 4.返回结果（Document 对象，包含 .page_content 和 .metadata）
results=retriever.invoke(query)
print(results)