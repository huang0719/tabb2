"""
Claude Messages API 路由 (/v1/messages)
为 Claude Code 提供 Anthropic Messages API 兼容端点。
"""

import json
import time
import uuid
import math
import logging

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from core.config import ConfigManager
from core.tabbit_client import TabbitClient
from core.token_manager import TokenManager
from core.log_store import LogStore, LogEntry
from core.tool_runtime import (
    execute_tool_call,
    format_tool_result_message,
    is_write_file_tool,
    extract_write_path,
    build_file_generation_prompt,
    extract_generated_file_content,
    build_builtin_file_content,
)
from core.claude_compat import (
    random_trigger_signal,
    map_claude_to_content,
    normalize_blocks,
    estimate_tokens,
    ToolifyParser,
    ClaudeSSEWriter,
    parse_toolified_text,
    events_to_content_blocks,
    build_direct_tool_call,
)

logger = logging.getLogger("tabbit2openai")

router = APIRouter()

_tm: TokenManager | None = None
_cfg: ConfigManager | None = None
_logs: LogStore | None = None
_fallback_clients: dict[str, TabbitClient] = {}

# Claude 模型名 → Tabbit 模型名映射
CLAUDE_MODEL_MAP = {
    "claude-opus-4-6": "best",
    "claude-sonnet-4-6": "best",
    "claude-sonnet-4-5": "best",
    "claude-haiku-4-5": "best",
    "claude-3-5-sonnet": "best",
    "claude-3-5-haiku": "best",
}


# // 初始化 Claude 路由共享状态
def init(token_manager: TokenManager, config: ConfigManager, log_store: LogStore):
    global _tm, _cfg, _logs
    _tm = token_manager
    _cfg = config
    _logs = log_store


# // 将 Claude 请求模型映射为 Tabbit 当前展示模型名
async def _resolve_tabbit_model(client: TabbitClient, model: str) -> str:
    """将请求中的模型名映射到 Tabbit 模型"""
    model_map = await client.get_model_map()
    key = model.lower()
    if key in model_map:
        return model_map[key]

    dash_key = key.replace("-", ".")
    if dash_key in model_map:
        return model_map[dash_key]

    # Claude 模型名映射
    for prefix, target in CLAUDE_MODEL_MAP.items():
        if model.startswith(prefix):
            if target in model_map:
                return model_map[target]
            break

    # 从 config 中读取默认模型
    default = _cfg.get("claude", "default_model") if _cfg else None
    if default and default.lower() in model_map:
        return model_map[default.lower()]
    if "default" in model_map:
        return model_map["default"]
    raise Exception(f"unsupported Claude model: {model}")


# // 按请求授权信息获取可用 Tabbit 客户端
async def _get_client_and_token(
    request: Request,
) -> tuple[TabbitClient, str, str]:
    """获取客户端实例，返回 (client, token_name, token_id)"""
    # 验证客户端 API key
    api_key = _cfg.get("proxy", "api_key") if _cfg else ""
    auth_header = request.headers.get("x-api-key") or request.headers.get(
        "authorization", ""
    )
    bearer = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else auth_header

    if _tm and _tm.has_tokens:
        if api_key and bearer != api_key:
            raise HTTPException(status_code=401, detail="invalid api key")
        token_info, client = await _tm.get_next()
        if token_info is None:
            raise HTTPException(
                status_code=503, detail="no available tokens (all cooling down)"
            )
        return client, token_info.get("name", "unknown"), token_info["id"]

    # fallback
    token = bearer
    if not token:
        raise HTTPException(status_code=401, detail="missing token")
    if token not in _fallback_clients:
        _fallback_clients[token] = TabbitClient(
            token,
            _cfg.get("tabbit", "base_url") if _cfg else None,
            _cfg.get("tabbit", "client_id") if _cfg else None,
            _cfg.get("tabbit", "proxy_url") if _cfg else None,
        )
    return _fallback_clients[token], "bearer", ""


# // 估算 Claude 请求输入 token 数
def _estimate_input_tokens(body: dict) -> int:
    """估算输入 token 数"""
    total_text = ""
    # system
    system = body.get("system")
    if system:
        if isinstance(system, str):
            total_text += system
        elif isinstance(system, list):
            for b in system:
                if isinstance(b, dict):
                    total_text += b.get("text", "")
    # messages
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            total_text += content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total_text += block.get("text", "")
                    total_text += block.get("thinking", "")
                    total_text += str(block.get("content", ""))
    # tools
    tools = body.get("tools", [])
    if tools:
        total_text += json.dumps(tools, ensure_ascii=False)

    return estimate_tokens(total_text)


# // 将 Tabbit 流式响应转换为 Claude Messages SSE
async def _stream_claude_response(
    client: TabbitClient,
    session_id: str,
    content: str,
    tabbit_model: str,
    body: dict,
    token_name: str,
    token_id: str,
):
    """流式生成 Claude SSE 响应"""
    request_id = uuid.uuid4().hex[:12]
    model = body.get("model", "claude-proxy")
    input_tokens = _estimate_input_tokens(body)

    writer = ClaudeSSEWriter(request_id, model, input_tokens)

    # 解析器配置
    tools = body.get("tools", [])
    has_tools = len(tools) > 0
    trigger_signal = body.get("_trigger_signal")  # 在调用前注入
    thinking_enabled = (
        body.get("thinking", {}).get("type") == "enabled"
        if isinstance(body.get("thinking"), dict)
        else False
    )
    parser = ToolifyParser(trigger_signal, thinking_enabled)

    # message_start
    yield writer.init_event()

    start_time = time.time()
    error_msg = ""

    try:
        async for event in client.send_message(session_id, content, tabbit_model):
            et = event["event"]
            ed = event["data"]

            if et == "message_chunk" and "content" in ed:
                text = ed["content"]
                for char in text:
                    parser.feed_char(char)
                    events = parser.consume_events()
                    if events:
                        for line in writer.handle_events(events):
                            yield line
            elif et in ("message_finish", "finish"):
                break

        # 流结束
        parser.finish()
        final_events = parser.consume_events()
        if final_events:
            for line in writer.handle_events(final_events):
                yield line

        if token_id and _tm:
            _tm.report_success(token_id)

    except Exception as e:
        error_msg = str(e)
        if token_id and _tm:
            _tm.report_error(token_id)
        # 尝试发送错误后仍然关闭流
        parser.finish()
        final_events = parser.consume_events()
        if final_events:
            for line in writer.handle_events(final_events):
                yield line
    finally:
        duration = time.time() - start_time
        if _logs:
            _logs.add(
                LogEntry(
                    model=body.get("model", "unknown"),
                    token_name=token_name,
                    stream=True,
                    status="success" if not error_msg else "error",
                    duration=duration,
                    error=error_msg,
                )
            )


# // 直接返回 Claude tool_use 流式响应
async def _stream_direct_tool_response(body: dict, direct_call: dict):
    request_id = uuid.uuid4().hex[:12]
    model = body.get("model", "claude-proxy")
    input_tokens = _estimate_input_tokens(body)
    writer = ClaudeSSEWriter(request_id, model, input_tokens)
    yield writer.init_event()
    for line in writer.handle_events(
        [{"type": "tool_call", "call": direct_call}, {"type": "end"}]
    ):
        yield line


# // 直接返回 Claude 工具执行结果流
async def _stream_executed_tool_response(body: dict, direct_call: dict, result: dict):
    request_id = uuid.uuid4().hex[:12]
    model = body.get("model", "claude-proxy")
    input_tokens = _estimate_input_tokens(body)
    text = format_tool_result_message(result)
    writer = ClaudeSSEWriter(request_id, model, input_tokens)
    yield writer.init_event()
    for line in writer.handle_events(
        [{"type": "text", "content": text}, {"type": "end"}]
    ):
        yield line


# // 读取 Claude 请求最后一条用户文本
def _last_user_text(body: dict) -> str:
    for message in reversed(body.get("messages", [])):
        if message.get("role") == "user":
            return normalize_blocks(message.get("content", ""))
    return ""


# // 使用 Tabbit 生成缺失的写文件内容
async def _fill_generated_file_content(
    client: TabbitClient, body: dict, arguments: dict
) -> dict:
    user_text = _last_user_text(body)
    path = arguments.get("path") or arguments.get("file_path") or arguments.get("filepath")
    if not path:
        path = extract_write_path(user_text)
    if not path:
        raise Exception("WriteFile missing path")
    if arguments.get("content") is not None:
        return arguments

    tabbit_model = await _resolve_tabbit_model(client, body.get("model", "best"))
    session_id = await client.create_chat_session()
    prompt = build_file_generation_prompt(user_text, str(path))
    generated = ""
    async for event in client.send_message(session_id, prompt, tabbit_model):
        if event["event"] == "message_chunk":
            generated += event["data"].get("content", "")
    content = extract_generated_file_content(generated)
    if not content:
        content = build_builtin_file_content(user_text, str(path))
    if not content:
        raise Exception("generated file content is empty")
    filled = dict(arguments)
    filled["path"] = str(path)
    filled["content"] = content
    return filled


@router.post("/v1/messages")
# // 处理 Anthropic Messages 请求
async def claude_messages(request: Request):
    """Anthropic Messages API 兼容端点"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")

    # 获取客户端
    client, token_name, token_id = await _get_client_and_token(request)

    # 工具调用准备
    tools = body.get("tools", [])
    trigger_signal = random_trigger_signal() if tools else None
    body["_trigger_signal"] = trigger_signal

    direct_call = build_direct_tool_call(body)
    if direct_call:
        try:
            if is_write_file_tool(direct_call["name"]):
                direct_call["arguments"] = await _fill_generated_file_content(
                    client, body, direct_call["arguments"]
                )
            tool_result = execute_tool_call(
                direct_call["name"], direct_call["arguments"], _cfg
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        is_stream = body.get("stream", True)
        if is_stream:
            return StreamingResponse(
                _stream_executed_tool_response(body, direct_call, tool_result),
                media_type="text/event-stream",
                headers={
                    "cache-control": "no-cache",
                    "connection": "keep-alive",
                },
            )
        request_id = uuid.uuid4().hex[:12]
        input_tokens = _estimate_input_tokens(body)
        result_text = format_tool_result_message(tool_result)
        return {
            "id": f"msg_{request_id}",
            "type": "message",
            "role": "assistant",
            "model": body.get("model", "claude-proxy"),
            "content": [
                {
                    "type": "text",
                    "text": result_text,
                }
            ],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": estimate_tokens(result_text),
            },
        }

    # 模型映射
    try:
        tabbit_model = await _resolve_tabbit_model(client, body.get("model", "best"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 注入全局 Claude system prompt
    claude_system_prompt = _cfg.get("claude", "system_prompt") if _cfg else ""
    if claude_system_prompt:
        body["_injected_system_prompt"] = claude_system_prompt

    # 构建发送内容
    content = map_claude_to_content(body, trigger_signal)

    # 创建聊天会话
    try:
        session_id = await client.create_chat_session()
    except Exception as e:
        if token_id and _tm:
            _tm.report_error(token_id)
        if _logs:
            _logs.add(
                LogEntry(
                    model=body.get("model", "unknown"),
                    token_name=token_name,
                    stream=True,
                    status="error",
                    error=str(e),
                )
            )
        raise HTTPException(status_code=502, detail=str(e))

    # Claude Code 总是 stream
    is_stream = body.get("stream", True)
    if is_stream:
        return StreamingResponse(
            _stream_claude_response(
                client, session_id, content, tabbit_model, body, token_name, token_id
            ),
            media_type="text/event-stream",
            headers={
                "cache-control": "no-cache",
                "connection": "keep-alive",
            },
        )

    # 非流式（少见，但仍支持）
    request_id = uuid.uuid4().hex[:12]
    model = body.get("model", "claude-proxy")
    input_tokens = _estimate_input_tokens(body)
    full_text = ""
    start_time = time.time()
    error_msg = ""

    try:
        async for event in client.send_message(session_id, content, tabbit_model):
            if event["event"] == "message_chunk":
                full_text += event["data"].get("content", "")
        if token_id and _tm:
            _tm.report_success(token_id)
    except Exception as e:
        error_msg = str(e)
        if token_id and _tm:
            _tm.report_error(token_id)
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        duration = time.time() - start_time
        if _logs:
            _logs.add(
                LogEntry(
                    model=model,
                    token_name=token_name,
                    stream=False,
                    status="success" if not error_msg else "error",
                    duration=duration,
                    error=error_msg,
                )
            )

    output_tokens = estimate_tokens(full_text)
    thinking_enabled = (
        body.get("thinking", {}).get("type") == "enabled"
        if isinstance(body.get("thinking"), dict)
        else False
    )
    events = parse_toolified_text(full_text, trigger_signal, thinking_enabled)
    content_blocks, stop_reason = events_to_content_blocks(events)
    if not content_blocks:
        raise HTTPException(status_code=502, detail="Claude response finished without content")
    return {
        "id": f"msg_{request_id}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


@router.post("/v1/messages/count_tokens")
# // 处理 Anthropic token 计数请求
async def count_tokens(request: Request):
    """Token 计数端点"""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON body")

    input_tokens = _estimate_input_tokens(body)
    return {"input_tokens": input_tokens}
