#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: ZackFair
# @Desc:
# @File: app.py
# @Date: 2025/5/15 21:47
import json
import asyncio
from typing import List, Dict

from openai import OpenAI
from fastmcp import Client


class MCPClient:
    def __init__(self, script: str, model="qwen3:0.6b"):
        self.script = script
        self.model = model

        self.client = OpenAI(
            api_key="None",
            base_url="http://127.0.0.1:11434/v1"
        )

        self.session = Client(script)
        self.tools = []

    async def prepare_tools(self):
        tools = await self.session.list_tools()
        self.tools = [
            {
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
            }
            for tool in tools
        ]

    async def chat(self, messages: List[Dict]):
        if not self.tools:
            await self.prepare_tools()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
        )
        if response.choices[0].finish_reason != 'tool_calls':
            return response.choices[0].message

        # 调用工具
        for tool_call in response.choices[0].message.tool_calls:
            result = await self.session.call_tool(tool_call.function.name, json.loads(tool_call.function.arguments))

            messages.append({
                'role': 'assistant',
                'content': result[0].text
            })

            return await self.chat(messages)

        return response

    async def loop(self):
        while True:
            async with self.session:
                question = input("User: ")
                response = await self.chat([
                    {
                        "role": "user",
                        "content": question,
                    }
                ])
                print(f"AI: {response.content}")


async def main():
    mcp_client = MCPClient("server.py")
    await mcp_client.loop()


if __name__ == '__main__':
    asyncio.run(main())


