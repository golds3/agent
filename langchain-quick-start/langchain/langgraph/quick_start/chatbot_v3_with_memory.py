"""
在chatbot_with_search_tool的基础上，添加历史记录能力，
基于graph的persistent checkpointing实现，
对于同一个thread_id,graph会自动缓存每一个step的state信息
因为缓存的是一个个state信息，所以相比于直接使用langchain中的memory组件（一般只缓存了对话信息），会更强大
"""


from typing import Annotated

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver



class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

"""
👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆
上面和chatbot_with_search_tool 一样
"""

# 基于内存的存储checkpoint
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

"""
测试一下效果
"""
# 以thread_id 的维度进行记忆存储
config = {"configurable": {"thread_id": "1"}} 

user_input = "Hi there! My name is Will."

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()

user_input = "我是谁?"
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()

user_input = "我是谁?"
config = {"configurable": {"thread_id": "2"}} 

events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()    
"""==============================================="""

def stream_graph_updates(user_input: str,seesion_id:str):
    config = {"configurable": {"thread_id": seesion_id}} 
    events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()
    # for event in graph.stream({"messages": [{"role": "user", "content": user_input}]},config,stream_mode="values"):
    #     for value in event:
    #         value["messages"][-1].pretty_print()
while True:
    try:
        sesson_id = input("conversaction session id: ")
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input,sesson_id)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input,sesson_id)
        break