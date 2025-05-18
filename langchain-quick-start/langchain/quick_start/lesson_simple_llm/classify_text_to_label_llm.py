import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
BASE_URL = "https://www.DMXapi.com/v1/"
os.environ["OPENAI_API_BASE"] = BASE_URL




from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

# 从文本中提取标注出这三个标注信息
"""
sentiment 情感倾向 积极、消极 ……
aggressiveness 攻击性、对抗性程度
"""
class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    aggressiveness: int = Field(
        description="How aggressive the text is on a scale from 1 to 10"
    )
    language: str = Field(description="The language the text is written in")


# LLM
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini").with_structured_output(
    Classification
)

inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
prompt = tagging_prompt.invoke({"input": inp})
response = llm.invoke(prompt)

print(f"标注结果：{response}")


# 对Classification进行更精细化的说明

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'ClassificationFiner' function.

Passage:
{input}
"""
)

class ClassificationFiner(BaseModel):
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"])
    aggressiveness: int = Field(
        ...,
        description="describes how aggressive the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(
        ..., enum=["spanish", "english", "french", "german", "italian","chinese"]
    )

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini").with_structured_output(
    ClassificationFiner
)    

inp = "Weather is ok here, I can go outside without much more than a coat"
prompt = tagging_prompt.invoke({"input": inp})
response = llm.invoke(prompt)

print(f"标注结果：{response}")

inp = "你的做法太令我失望了，我对你失去了信心"
prompt = tagging_prompt.invoke({"input": inp})
response = llm.invoke(prompt)

print(f"标注结果：{response}")