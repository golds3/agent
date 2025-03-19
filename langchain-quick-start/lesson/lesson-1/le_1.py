# befor run ,source langchain-quick-start/bin/activate
from dotenv import load_dotenv
import os

load_dotenv()  # 加载 .env 文件中的环境变量
DEEKSEEP_API_KEY = os.getenv("DEEKSEEP_API_KEY")
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 初始化模型 
# deepseek 目前api不完善，先用本地ollama
# from langchain_deepseek import ChatDeepSeek


# llm = ChatDeepSeek(
#     model="deepseek-chat"
#     ,api_key = DEEKSEEP_API_KEY)

from langchain_ollama import OllamaLLM 
llm = OllamaLLM(model="deepseek-r1:7b")


# 测试模型
response = llm.invoke("how can langsmith help with testing?")
# print("Response from llm.invoke:", response)  # 添加 print 语句

# 创建提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])

# 创建输出解析器
output_parser = StrOutputParser()

# 创建LLM链
chain = prompt | llm | output_parser

# 调用链
result = chain.invoke({"input": "how can langsmith help with testing?"})
# print("Response from chain.invoke:", result)  # 添加 print 语句

"""
上面的llm是回答不了how can langsmith help with testing? 这个问题的，所以要给LLM提高上下文信息，比如langsmith的用户手册
"""
# 检索链
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()
# 引入嵌入模型
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="deepseek-r1:7b")

# 引入向量数据库
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

from langchain.chains.combine_documents import create_stuff_documents_chain
# 设置prompt 回答问题只从提供的上下文去回答
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
<context>
{context}
</context>
 
Question: {input}""")
document_chain = create_stuff_documents_chain(llm, prompt)

# 方式一 直接把上下文："langsmith can let you visualize test results" 传入
from langchain_core.documents import Document
response =document_chain.invoke({
    "input": "how can langsmith help with testing?",
    "context": [Document(page_content="langsmith can let you visualize test results")]
})
# print("Response from document_chain.invoke",response)
# 方式二 使用检索器动态选择最相关的文档并将其传递给给定的问题
from langchain.chains import create_retrieval_chain
 
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
# print("Response from retrieval_chain.invoke",response)
 



# 对话检索链
"""
上面的llm chain 只能回答单个问题，现在构建一个可以进行对话的llm chain
原理就是在 检索链 create_retrieval_chain 中，要把对话的信息也记录下来
"""
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
# 首先，我们需要一个可以传递给LLM来生成此搜索查询的提示
 
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "根据上面的对话，生成一个搜索查询来获取与对话相关的信息")
])
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

from langchain_core.messages import HumanMessage, AIMessage
 
chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
# 假设之间对话问过能不能help，现在直接问how ，看看能不能记得这个对话历史
response = retriever_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})
# print("response depend on chat-history:",response)




prompt = ChatPromptTemplate.from_messages([
    ("system", "根据下面的上下文回答用户的问题：\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(llm, prompt)
 
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
retrieval_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})
print("response depend on chat-history:",response)
