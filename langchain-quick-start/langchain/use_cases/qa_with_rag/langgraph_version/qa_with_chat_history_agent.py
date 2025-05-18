"""
基于Agent 实现，有llm自己决定调用哪些tool
不用我们手动编排顺序
"""
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
import bs4



llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# LSEVR流程
# 第一步 load 加载文档
loader = WebBaseLoader(
  web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
  bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
  )
)
docs = loader.load()

# 第二步 Spilit 分割文档

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)


# 第三步 Embeddings 实例化一个嵌入器
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# 第四步 Vector 实例化一个向量存储
vector_store = InMemoryVectorStore(embeddings)
# 存储
_ = vector_store.add_documents(documents=all_splits)



# 自定义构建一个tool来执行 r-retrieval 检索这一步骤
@tool(response_format="content_and_artifact")
def retrieve(query:str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized,retrieved_docs

memory = MemorySaver()

agent_executor = create_react_agent(llm, tools=[retrieve], checkpointer=memory)


config = {"configurable": {"thread_id": "def234"}}

input_message = (
    "What is the standard method for Task Decomposition?\n\n"
    "Once you get the answer, look up common extensions of that method."
)

for event in agent_executor.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
    config=config,
):
    event["messages"][-1].pretty_print()