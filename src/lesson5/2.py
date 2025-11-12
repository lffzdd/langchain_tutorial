import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.utils.math import cosine_similarity
import numpy as np

os.environ["OPENAI_API_KEY"] = "sk-ljuovztdpgqafjnufoxhqbuzfzyvctualkzjxekycvpywxpx"

emb = OpenAIEmbeddings(model="BAAI/bge-m3", base_url="https://api.siliconflow.cn/v1")

# 嵌入查询和文档
q_vec = emb.embed_query("query:NLP中什么比较普遍")
d_vecs = emb.embed_documents(
    [
        "passage:使用BGE并进行向量归一化有帮助",
        "passage:深度学习模型在NLP中很流行",
        "passage:向量数据库用于高效检索",
    ]
)

print(f"查询向量维度: {len(q_vec)}")
print(f"文档数量: {len(d_vecs)}, 每个文档向量维度: {len(d_vecs[0])}")
print("文档向量示例:", q_vec[:10])  # 打印前10个元素作为示例

scores=cosine_similarity([q_vec],d_vecs)[0]
print(scores)

q_vec = np.array(q_vec)
q_vec=q_vec/np.linalg.norm(q_vec)
d_vecs = np.array(d_vecs)
d_vecs=d_vecs/np.linalg.norm(d_vecs, axis=1, keepdims=True)

print("归一化后的查询向量:", q_vec[:10])  # 打印前10个元素作为示例

scores = np.dot(q_vec, d_vecs.T)
print(scores)