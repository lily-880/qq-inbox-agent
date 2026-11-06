"""DeepSeek（OpenAI 兼容）客户端。密钥只从环境变量 / .env 读取。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .llm_types import LLMResponse, ToolCall

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


def get_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "缺少 DEEPSEEK_API_KEY。请在项目根目录创建 .env（参考 .env.example）。"
        )
    base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).strip()
    return OpenAI(api_key=api_key, base_url=base_url)


def get_model() -> str:
    return os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


class DeepSeekLLM:
    """在线后端。和 StubLLM 用同一个 complete() 形状，方便 agent 循环复用。"""

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse:
        client = get_client()
        kwargs: dict[str, Any] = {
            "model": get_model(),
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        elif json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        message = resp.choices[0].message
        calls: list[ToolCall] = []
        for tool_call in message.tool_calls or []:
            raw_args = tool_call.function.arguments or "{}"
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed = {}
            if not isinstance(parsed, dict):
                parsed = {}
            calls.append(
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=parsed,
                )
            )
        return LLMResponse(content=message.content, tool_calls=calls)


def chat_complete(messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
    """V1 分类评测仍走这个入口：只要文本，不要工具。"""
    response = DeepSeekLLM().complete(messages, temperature=temperature)
    if not response.content:
        raise RuntimeError("模型返回空内容")
    return response.content
