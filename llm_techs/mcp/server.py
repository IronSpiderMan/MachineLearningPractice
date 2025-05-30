#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: ZackFair
# @Desc:
# @File: server.py
# @Date: 2025/5/15 21:11


"""
1. 初始化fastmcp
2. 创建一个函数，文档，功能描述、参数描述、返回值描述
3. 使用@tool 注解
3. 启动 mcp 服务器
"""
from datetime import datetime

from fastmcp import FastMCP

mcp = FastMCP()


@mcp.tool()
def get_today():
    """
    获取今天的时间，精确到秒
    :return: 年月日时分秒的字符串
    """
    return datetime.today().strftime('%Y.%m.%d %H-%M-%S')


@mcp.tool()
def get_weather(city: str, date: str):
    """
    获取 city 的天气情况
    :param city: 城市
    :param date: 日期
    :return: 城市天气情况的描述
    """
    return f"{city} {date} 天晴，18度。"


if __name__ == '__main__':
    mcp.run()
