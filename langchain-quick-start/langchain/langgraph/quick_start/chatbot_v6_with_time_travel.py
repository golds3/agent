"""
之前版本构建的chatbot都是一个线性的机器人
即总是从grapy的start开始，-->human ask1 --->ai response1---->human ask2 --->ai response2--> ...

然后我们有时候想回到之前的某次问答，从这里开始对话 ，类似 create branch of (here)                            
                              
基于graph的get_state_history函数实现这个功能，因为graph的核心就是state的流转，
而我们使用了memory checkpoint保存了历史信息,所以可以依靠历史信息找到我们想要回到的state处
"""

from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


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
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

"""跟chatbot进行几轮对话"""
config = {"configurable": {"thread_id": "1"}}
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm learning LangGraph. "
                    "Could you do some research on it for me?"
                ),
            },
        ],
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Ya that's helpful. Maybe I'll "
                    "build an autonomous agent with it!"
                ),
            },
        ],
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()        

"""看看目前为止graph的历史state信息，进行了两组问答，
graph的路程 start--->chatbot-->tools--->chatbot--->()---->start--->chatbot-->()"""        
to_replay = None
for state in graph.get_state_history(config):
    print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
    print("-" * 80)
    # 回到第二次对话，graph准备把human message发送给chatbot的state
    if len(state.values["messages"]) == 5:
        # We are somewhat arbitrarily selecting a specific state based on the number of chat messages in the state.
        to_replay = state
        
print(to_replay.next) # 现在我们回到了第二次对话 准备把human message发送给chatbot的state
print(to_replay.config)
# 修改提问问题，让ai去调用tools查询
# The `checkpoint_id` in the `to_replay.config` corresponds to a state we've persisted to our checkpointer.
for event in graph.stream(    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Ya that's helpful. Maybe I'll "
                    "build an autonomous agent with it! Please help me search what show i know before can build an autonomous agent"
                ),
            },
        ],
    }, to_replay.config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()







