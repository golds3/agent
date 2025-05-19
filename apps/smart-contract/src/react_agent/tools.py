"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""
import pymysql

import asyncio
from typing import Annotated, Any, Callable, List, Optional, cast

from langchain_tavily import TavilySearch  # type: ignore[import-not-found]

from react_agent.configuration import Configuration
from react_agent.agents.rules_agent import RulesAgent
from langchain_core.tools import InjectedToolCallId
from langchain_core.messages import ToolMessage,HumanMessage
from langgraph.types import Command 

async def search(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_context()
    wrapped = TavilySearch(max_results=configuration.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


async def rules_tool(query: str,tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    查询合约规则工具，根据传入的 query 返回相关规则文本。

    这个函数内部会自动创建 RulesAgent 实例并查询规则。
    """
    rules_agent = RulesAgent("/Users/ham/Desktop/project/ai/agent/apps/smart-contract/resources")
    resp = rules_agent.query_rules(query)
    if "服务外包" in query:
        contract_type_enum = 2
    elif "采购合同" in query:
        contract_type_enum = 0
    else:
        contract_type_enum = 1    

    state_update = {
        "contract_type_enum": contract_type_enum,
        "messages": [ToolMessage(resp, tool_call_id=tool_call_id),HumanMessage("合同的历史数据")],
    }
    # We return a Command object in the tool to update our state.
    return Command(update=state_update)



# 你的数据库配置，替换成真实信息
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "ai",
    "port": 3306,
    "charset": "utf8mb4",
}


async def history_tool(
    tool_call_id: Annotated[str, InjectedToolCallId],
    contract_type_enum: int,
    limit: int = 5,
) -> Command:
    """
    历史合约查询工具，根据 state 里的 contract_type_enum 查询历史合约，
    并返回查询结果，写入 messages 更新 state。
    """
    print(f'查询类型:',contract_type_enum)

    def query_db(contract_type: int, limit: int):
        conn = pymysql.connect(**DB_CONFIG)
        try:
            with conn.cursor() as cursor:
                sql = """
                    SELECT *
                    FROM contract_meta WHERE field_value=%s ORDER BY created_at DESC LIMIT %s
                """
                cursor.execute(sql, (contract_type, limit))
                rows = cursor.fetchall()
                return rows
        finally:
            conn.close()

    rows = await asyncio.to_thread(query_db, contract_type_enum, limit)

    if not rows:
        resp_text = f"未找到 contract_type={contract_type_enum} 的历史合约记录。"
    else:
        # 简单格式化输出
        lines = []
        for r in rows:
            line = (
                f"id: {r[0]}, 合约类型: {r[1]}, 合约级别: {r[2]}, 合约规则: {r[3]}, 创建时间: {r[4]}"
            )
            lines.append(line)
        resp_text = "查询到的历史合约：\n" + "\n".join(lines)

    state_update = {
        "messages": [ToolMessage(resp_text, tool_call_id=tool_call_id)],
    }
    return Command(update=state_update)


TOOLS: List[Callable[..., Any]] = [search,rules_tool,history_tool]
