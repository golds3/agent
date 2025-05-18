"""
在chatbot_base的基础上，在graph中加入搜索工具 ---Tavily Search Engine
"""


from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
import json

from langchain_core.messages import ToolMessage



class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

llm = init_chat_model("gpt-4o-mini", model_provider="openai")



def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

# 遍历chatbot中的messages列表，当前用户输入存储，打印AI的最新响应["messages"][-1]，
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            message = value["messages"][-1]
            # 检查消息是否包含工具调用
            if hasattr(message, "tool_calls") and message.tool_calls:
                print("Assistant: [Tool Call]", message)
            else:
                print("Assistant:", message.content)

"""
👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆👆
上面和chatbot_base 一样
"""

from langchain_community.tools.tavily_search import TavilySearchResults
# 要先去申请一下key，https://app.tavily.com/home  ，然后export TAVILY_API_KEY
tool = TavilySearchResults(max_results=2)
tools = [tool]
# Modification: tell the LLM which tools it can call
llm_with_tools = llm.bind_tools(tools)



"""
tools节点，提供call能力,调用TavilySearchResults 
"""
class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


tool_node = BasicToolNode(tools=[tool])

"""也可以直接使用框架的ToolNode模板"""
# tool_node = ToolNode(tools=[tool])


graph_builder.add_node("tools", tool_node)


"""
Use in the conditional_edge to route to the ToolNode if the last message
has tool calls. Otherwise, route to the end.
"""
def route_tools(
    state: State,
):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

"""conditional_edges 条件型edge，条件性连接两个节点"""
# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"

    # 这里的"tools"就是route_tools根据条件返回的route_tools
    {"tools": "tools", END: END},
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break