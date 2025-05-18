import getpass
import os

# 因为我使用的是中转api，所以需要更改一下base_url
BASE_URL = "https://www.DMXapi.com/v1/"
os.environ["OPENAI_API_BASE"] = BASE_URL

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")


from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

response = model.invoke(messages)
print(f"翻译结果: {response.content}")


"""
接下来使用prompt template，动态设置翻译的语言，和用户的输入文本
"""

from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
prompt = prompt_template.invoke({"language": "Chinese", "text": "hi!"})
print(f"提示词：{prompt}")
response = model.invoke(prompt)
print(f"翻译结果: {response.content}")