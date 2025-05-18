"""
这个版本的机器人只是具备最基础的对话功能
主要是了解langgraph的使用方式
1.StateGraph  --- 为机器人定义状态机，状态机决定了机器人能调用的模型和函数，以及制定机器人状态转变的规则
"""
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langchain.chat_models import init_chat_model
from langchain_anthropic import ChatAnthropic


llm = init_chat_model("gpt-4o-mini", model_provider="openai")
# llm = ChatAnthropic(model="claude-3-5-sonnet-20240620",api_key="",base_url="https://www.dmxapi.com")

"""
定义一个Graph的时候，一般先定义一个State对象，用来定义 Graph的处理状态更新的模式和函数
"""
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

"""
这个graph 具备 状态转移和添加messages 这两个功能
"""
graph_builder = StateGraph(State)





"""
每个graph节点都是一个function
这个结点定义的规则如下： 接受一个状态，返回一个字典，包含所有的消息列表
"""
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
"""向Graph中添加第一个节点"""
graph_builder.add_node("chatbot", chatbot)

"""
像图一样，为刚刚构建的chatbot节点添加一个边，START 表面这是这个graph的入口点
"""
graph_builder.add_edge(START, "chatbot")
# 同样，添加一个END 作为graph的出口
graph_builder.add_edge("chatbot", END)
# 建造出一个graph
graph = graph_builder.compile()


# 遍历chatbot中的messages列表，当前用户输入存储，打印AI的最新响应["messages"][-1]，
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

# 现在可以使用构建的graph进行chatbot的调用了
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