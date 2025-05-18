from langchain_core.documents import Document


from langchain_community.document_loaders import PyPDFLoader

"""
流程：加载文档--> 文档分割 --->向量嵌入--->存储向量数据--->使用检索器检索
"""
# 加载文档
file_path = "./resource/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))


"""
page_content :文档内容
metadata：存储与文档内容相关的额外信息，用于文档检索的时候进行数据过滤
"""
print(f"{docs[0].page_content[:200]}\n")
print(docs[0].metadata)

# 文档分割
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(f"分割文档个数：{len(all_splits)}")

# 向量嵌入

import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import OpenAIEmbeddings
BASE_URL = "https://www.DMXapi.com/v1/"
os.environ["OPENAI_API_BASE"] = BASE_URL
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])

# 向量存储
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(documents=all_splits)


# 生成检索器（
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)


# 检索文档
result = retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)

print(f"检索结果：{result}")