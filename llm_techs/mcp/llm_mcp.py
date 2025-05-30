#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: ZackFair
# @Desc: 
# @File: llm_mcp.py
# @Date: 2025/5/19 13:42
"""
1. mcp客户端
2. OpenAI客户端
3. 将 MCP 工具转换成 OpenAI 可用形式
4. 在调用 OpenAI 接口时，传入工具
5. 如果响应的是 call_tools，则遍历工具，调用工具

"""
import asyncio
import json

from fastmcp import Client
from openai import OpenAI

mcp_client = Client('server.py')
openai_client = OpenAI(
    base_url='http://127.0.0.1:11434/v1',
    api_key="None"
)


async def main():
    async with mcp_client:
        tools = await mcp_client.list_tools()
        """
        openai tool要求：
        {
            "type": "function",
            "function": {
                "name": "",
                "description": "",
                "input_schema": {}
            }
        }
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                }
            }
            for tool in tools
        ]
        response = openai_client.chat.completions.create(
            model='qwen3:0.6b',
            messages=[
                {
                    'role': 'user',
                    'content': '今天北京天气怎么样？'
                }
            ],
            tools=tools
        )
        if response.choices[0].finish_reason != 'tool_calls':
            print(response)
        else:
            for tool_call in response.choices[0].message.tool_calls:
                tool_result = await mcp_client.call_tool(tool_call.function.name, json.loads(tool_call.function.arguments))
                print(tool_result)


if __name__ == '__main__':
    asyncio.run(main())
