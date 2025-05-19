"""
从本地读取所有合约规则文件
"""
import asyncio
import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RulesAgent:
    def __init__(self, rules_dir:str):
        self.rules_dir = rules_dir
        self.llm = ChatOpenAI(model_name="gpt-4.1-nano")
        # self._load_and_index_documents_async()
        self._load_and_index_documents()

    async def _load_and_index_documents_async(self):
        return await asyncio.to_thread(self._load_and_index_documents)
    def _load_and_index_documents(self):
        documents = []
        """加载本地文件"""
        for filename in os.listdir(self.rules_dir):
            path = os.path.join(self.rules_dir, filename)
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif filename.endswith(".docx"): # 暂不支持doc文件
                loader = Docx2txtLoader(path)    
            else:
                continue
            documents.extend(loader.load())    
        """分词"""
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(documents)   
        """存储"""
        embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(split_docs, embeddings)

    def query_rules(self, query: str):
        docs = self.vectorstore.similarity_search(query)
        # 你可以用 llm 来做后续处理，比如总结答案
        # 这里简单返回文档文本
        return "\n\n".join([doc.page_content for doc in docs])


