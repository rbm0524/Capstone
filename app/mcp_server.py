# app/mcp_server.py

import os
from notion_client import Client

# MCP 등록용 데코레이터와 레지스트리
MCP_REGISTRY = {}

def mcp(func):
    """MCP 함수 등록용 데코레이터"""
    MCP_REGISTRY[func.__name__] = func
    return func

@mcp
def save_to_notion(summary_text: str):
    """
    사용자의 대화 내용을 요약해서 Notion에 기록합니다.
    """
    notion_token = os.getenv("NOTION_API_KEY")
    notion_db_id = os.getenv("NOTION_DB_ID")
    notion = Client(auth=notion_token)

    notion.pages.create(
    parent={"database_id": notion_db_id},
    properties={
        "이름": {"title": [{"text": {"content": "요약 기록"}}]},
        "Summary": {"rich_text": [{"text": {"content": summary_text}}]},
    }
)
    return "노션에 요약 내용을 성공적으로 기록했습니다."


# 추가된 MCP 에이전트와 통신용 설정
import asyncio
import subprocess

class StdioServerParameters:
    def __init__(self, command, args=None):
        self.command = command
        self.args = args or []

async def stdio_client(server_params):
    process = await asyncio.create_subprocess_exec(
        server_params.command, *server_params.args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE
    )

    async def read():
        line = await process.stdout.readline()
        return line.decode()

    async def write(data: str):
        process.stdin.write(data.encode())
        await process.stdin.drain()

    class StdioClientContext:
        async def __aenter__(self):
            return read, write

        async def __aexit__(self, exc_type, exc, tb):
            process.terminate()

    return StdioClientContext()


# 간단한 ClientSession 클래스 (필요 시 확장 가능)
class ClientSession:
    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer

    async def initialize(self):
        # 예시 초기화. 실제로는 handshake 등이 들어갈 수 있음
        await self.writer("READY\n")

    async def call_tool(self, tool_name: str, args: list):
        call_str = f"{tool_name} {' '.join(args)}\n"
        await self.writer(call_str)
        response = await self.reader()
        return response
