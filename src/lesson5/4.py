from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # 每个块的最大字符数
    chunk_overlap=13,  # 每个块之间的重叠字符数
    length_function=len,
)

docs = [
    "passage:使用BGE并进行向量归一化有帮助"
    "使用BGE并进行向量归一化有帮助1"
    "使用BGE并进行向量归一化有帮助2"
    "使用BGE并进行向量归一化有帮助3"
    "使用BGE并进行向量归一化有帮助4"
    "使用BGE并进行向量归一化有帮助5"
    "使用BGE并进行向量归一化有帮助6",
    "passage:深度学习模型在NLP中很流行",
    "passage:向量数据库用于高效检索",
]
# docs转为Document
docs = text_splitter.create_documents(docs)

for i,doc in enumerate(docs):
    print(f"文档块 {i + 1} 内容:\n{doc}\n")