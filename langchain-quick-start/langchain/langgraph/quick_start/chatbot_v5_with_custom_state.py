"""
langgraph 的每一个节点都需要遵循 state定义的规范。--按照state的结构进行输入输出
前面简单的使用了messages list作为state规范

这里我们自定义一个更加复杂的state
实现功能：
chatbot使用search tool进行信息查询，把查询的信息发给human审查
"""


from typing import Annotated
from langchain.chat_models import init_chat_model

from typing_extensions import TypedDict

from langgraph.graph.message import add_messages

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool

from langgraph.types import Command, interrupt

"""state 定义了name和birthday结构，每一个node需要构建和传输这个结构"""
class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str




@tool
# Note that because we are generating a ToolMessage for a state update, we
# generally require the ID of the corresponding tool call. We can use
# LangChain's InjectedToolCallId to signal that this argument should not
# be revealed to the model in the tool's schema.
def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    # If the information is correct, update the state as-is.
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    # Otherwise, receive information from the human reviewer.
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    # This time we explicitly update the state with a ToolMessage inside
    # the tool.
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    # We return a Command object in the tool to update our state.
    return Command(update=state_update)


"""
构建graph
"""
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
tool = TavilySearchResults(max_results=2)
tools = [tool, human_assistance]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

"""prompt  问问生日和名字"""
# 第一个问题会用到搜索工具，第二个指示AI使用human_assistance tool
user_input = (
    "My name is ham,Can you look up when i was born? "
    "When you have the answer, use the human_assistance tool for review."
)
config = {"configurable": {"thread_id": "1"}}

events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

"""
Tool Calls:
  human_assistance (call_j85iuM28E9vihwf0ynOq2ZDT)
 Call ID: call_j85iuM28E9vihwf0ynOq2ZDT
  Args:
    birthday:
    name: ham
    等待human给出回答
    
    可见ai通过查询工具查不到我的个人信息，
    所以我们告诉他答案
    最后给我们的答案就是我们纠正他的答案

    注意！对于我们human的response，ai只是用于验证，
    如果ai确信自己的回答说正确的，那么他不会采取我们给的信息
"""


human_command = Command(
    resume={
        "name": "ham",
        "birthday": "Jan 01,2001",
    },
)

events = graph.stream(human_command, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# 我们可以随时手动对state的值进行修改 ，
# 但是不建议，还是通过interrupt模式进行人机交互来更新state
graph.update_state(config, {"name": "LangGraph (library)"})
snapshot = graph.get_state(config)
print({k: v for k, v in snapshot.values.items() if k in ("name", "birthday")})        