import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.utils.math import cosine_similarity

os.environ["OPENAI_API_KEY"] = "sk-ljuovztdpgqafjnufoxhqbuzfzyvctualkzjxekycvpywxpx"

emb = OpenAIEmbeddings(model="BAAI/bge-m3", base_url="https://api.siliconflow.cn/v1")

# 嵌入查询和文档
q_vec = emb.embed_query("query:NLP")
d_vecs = emb.embed_documents(
    [
        "passage:使用BGE并进行向量归一化有帮助",
        "passage:深度学习模型在NLP中很流行",
        "passage:向量数据库用于高效检索",
    ]
)

print(f"查询向量维度: {len(q_vec)}")
print(f"文档数量: {len(d_vecs)}, 每个文档向量维度: {len(d_vecs[0])}")

scores=cosine_similarity([q_vec],d_vecs)[0]
print(scores)