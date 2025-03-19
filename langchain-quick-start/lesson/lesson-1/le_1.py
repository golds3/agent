# befor run ,source langchain-quick-start/bin/activate
from dotenv import load_dotenv
import os

load_dotenv()  # 加载 .env 文件中的环境变量
DEEKSEEP_API_KEY = os.getenv("DEEKSEEP_API_KEY")

# 初始化模型

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

 
llm = ChatDeepSeek(
    model="deepseek-chat"
    ,api_key = DEEKSEEP_API_KEY)

# 测试模型
response = llm.invoke("how can langsmith help with testing?")
print("Response from llm.invoke:", response)  # 添加 print 语句

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
print("Response from chain.invoke:", result)  # 添加 print 语句