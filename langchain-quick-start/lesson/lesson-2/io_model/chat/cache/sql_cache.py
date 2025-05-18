# 我们可以使用SQLite缓存做同样的事情
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

 
set_llm_cache(SQLiteCache(database_path=".langchain.db")) # 缓存会持久化到db


import time
import base
llm = base.getLLM()
 
# 第一次，它尚未在缓存中，所以需要更长的时间
start_time = time.time()  # 记录开始时间
resp = llm.invoke("告诉我一个笑话")
print(resp)
end_time = time.time()  # 记录结束时间
print(f"第一次执行时间: {end_time - start_time:.2f} 秒")  # 打印执行时间

# 第二次，由于已存在于缓存中，因此速度更快
start_time = time.time()  # 记录开始时间
resp = llm.invoke("特朗普是谁")
print(resp)
end_time = time.time()  # 记录结束时间
print(f"第二次执行时间: {end_time - start_time:.2f} 秒")  # 打印执行时间



# 第二次，由于已存在于缓存中，因此速度更快
start_time = time.time()  # 记录开始时间
resp = llm.invoke("告诉我一个笑话")
print(resp)
end_time = time.time()  # 记录结束时间
print(f"第二次执行时间: {end_time - start_time:.2f} 秒")  # 打印执行时间