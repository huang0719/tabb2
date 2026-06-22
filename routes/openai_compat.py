import json
import time
import uuid
import logging
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.tabbit_client import TabbitClient, fetch_model_map
from core.token_manager import TokenManager
from core.log_store import LogStore, LogEntry
from core.config import ConfigManager
from core.tool_runtime import (
    execute_tool_call,
    format_tool_result_message,
    is_write_file_tool,
    extract_write_path,
    build_file_generation_prompt,
    extract_generated_file_content,
    build_builtin_file_content,
)

logger = logging.getLogger("tabbit2openai")

router = APIRouter()

# 这些在 tabbit2api.py 中通过 app.state 注入
_tm: TokenManager | None = None
_cfg: ConfigManager | None = None
_logs: LogStore | None = None
_fallback_clients: dict[str, TabbitClient] = {}


# // 注入路由运行所需的共享状态
def init(token_manager: TokenManager, config: ConfigManager, log_store: LogStore):
    global _tm, _cfg, _logs
    _tm = token_manager
    _cfg = config
    _logs = log_store


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "best"
    messages: list[ChatMessage]
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None


# // 从 OpenAI 工具定义中读取工具名称
def _tool_name(tool: dict[str, Any]) -> str:
    function = tool.get("function")
    if not isinstance(function, dict):
        return ""
    return str(function.get("name", ""))


# // 从用户文本中提取工具参数
def _extract_tool_arguments(tool: dict[str, Any], user_text: str) -> dict[str, Any] | None:
    function = tool.get("function")
    if not isinstance(function, dict):
        return None
    parameters = function.get("parameters", {})
    if not isinstance(parameters, dict):
        return {}
    props = parameters.get("properties", {})
    required = parameters.get("required", [])
    if not isinstance(props, dict) or not isinstance(required, list):
        return {}

    args: dict[str, Any] = {}
    for name in props:
        lowered = str(name).lower()
        if lowered in ("path", "file_path", "filepath"):
            match = re.search(
                r"([A-Za-z]:\\[\s\S]+?)(?:[，,]\s*(?:内容|content)|。|；|;|$)",
                user_text,
            )
            if not match:
                match = re.search(r"创建文件\s*([\s\S]+?)(?:[，,]\s*内容|。|；|;|$)", user_text)
            if match:
                args[name] = match.group(1).strip()
        elif lowered in ("content", "text"):
            match = re.search(
                r"(?:内容(?:为|是)?|content\s*(?:is|=|:)?)[\s:：]*([^\n。；;]+)",
                user_text,
                re.I,
            )
            if not match:
                match = re.search(r"[，,]\s*内容\s*([\s\S]+?)(?:。|；|;|$)", user_text)
            if match:
                args[name] = match.group(1).strip()
        elif lowered in ("city", "location"):
            match = re.search(r"(?:查询|查看|获取)\s*([^\s，。；;]+?)\s*(?:当前)?(?:时间|天气)", user_text)
            if not match:
                match = re.search(r"(?:for|in)\s+([A-Za-z][A-Za-z\s-]*)", user_text, re.I)
            if match:
                args[name] = match.group(1).strip()

    missing = [name for name in required if name not in args]
    if missing:
        return None
    return args


# // 在明确工具请求时构造 OpenAI tool_calls
def _build_direct_tool_call(req: ChatCompletionRequest) -> dict[str, Any] | None:
    tools = req.tools or []
    if not tools:
        return None

    selected_tool = None
    forced_tool = False
    if isinstance(req.tool_choice, dict):
        if req.tool_choice.get("type") == "function":
            function = req.tool_choice.get("function")
            selected_name = function.get("name") if isinstance(function, dict) else None
            selected_tool = next(
                (tool for tool in tools if _tool_name(tool) == selected_name), None
            )
            forced_tool = selected_tool is not None
    elif req.tool_choice == "required":
        selected_tool = tools[0]
        forced_tool = True

    user_text = ""
    for message in reversed(req.messages):
        if message.role == "user":
            user_text = message.content
            break

    if not selected_tool:
        for tool in tools:
            name = _tool_name(tool)
            if name and name in user_text:
                selected_tool = tool
                break

    if not selected_tool and len(tools) == 1:
        lowered_text = user_text.lower()
        if (
            "调用" in user_text
            or "工具" in user_text
            or "call" in lowered_text
            or "tool" in lowered_text
            or (is_write_file_tool(_tool_name(tools[0])) and extract_write_path(user_text))
        ):
            selected_tool = tools[0]

    if not selected_tool:
        return None

    arguments = _extract_tool_arguments(selected_tool, user_text)
    if arguments is None:
        if is_write_file_tool(_tool_name(selected_tool)):
            path = extract_write_path(user_text)
            if path:
                arguments = {"path": path}
            elif forced_tool:
                arguments = {}
            else:
                return None
        elif forced_tool:
            arguments = {}
        else:
            return None

    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": _tool_name(selected_tool),
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


# // 返回 OpenAI 非流式工具调用响应
def _tool_call_response(req: ChatCompletionRequest, completion_id: str, tool_call: dict[str, Any]) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


# // 返回 OpenAI 流式工具调用响应
async def _stream_tool_call_response(req: ChatCompletionRequest, completion_id: str, tool_call: dict[str, Any]):
    yield (
        f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
    )
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["function"]["name"],
                                "arguments": tool_call["function"]["arguments"],
                            },
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    finish = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
    }
    yield f"data: {json.dumps(finish, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


# // 返回 OpenAI 非流式工具执行结果
def _tool_result_response(req: ChatCompletionRequest, completion_id: str, tool_call: dict[str, Any], result: dict[str, Any]) -> dict:
    content = format_tool_result_message(result)
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


# // 返回 OpenAI 流式工具执行结果
async def _stream_tool_result_response(req: ChatCompletionRequest, completion_id: str, tool_call: dict[str, Any], result: dict[str, Any]):
    content = format_tool_result_message(result)
    yield (
        f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
    )
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    finish = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(finish, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


# // 读取最后一条用户消息
def _last_user_text(req: ChatCompletionRequest) -> str:
    for message in reversed(req.messages):
        if message.role == "user":
            return message.content
    return ""


# // 使用 Tabbit 生成缺失的写文件内容
async def _fill_generated_file_content(
    client: TabbitClient, req: ChatCompletionRequest, arguments: dict[str, Any]
) -> dict[str, Any]:
    user_text = _last_user_text(req)
    path = arguments.get("path") or arguments.get("file_path") or arguments.get("filepath")
    if not path:
        path = extract_write_path(user_text)
    if not path:
        raise Exception("WriteFile missing path")
    if arguments.get("content") is not None:
        return arguments

    model_map = await client.get_model_map()
    tabbit_model = model_map.get(req.model.lower()) or model_map.get("default")
    if not tabbit_model:
        raise Exception(f"unsupported model: {req.model}")
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


# // 将 OpenAI 消息数组合并为 Tabbit 单条输入
def _build_content(messages: list[ChatMessage]) -> str:
    system_prompt = _cfg.get("proxy", "system_prompt") if _cfg else ""
    if len(messages) == 1 and not system_prompt:
        return messages[0].content
    parts = []
    if system_prompt:
        parts.append(f"[System]: {system_prompt}")
    for m in messages:
        label = {"user": "User", "assistant": "Assistant", "system": "System"}.get(
            m.role, m.role.capitalize()
        )
        parts.append(f"[{label}]: {m.content}")
    return "\n\n".join(parts) + "\n\n[Assistant]:"


# // 按请求授权信息获取可用 Tabbit 客户端
async def _get_client_and_token(
    authorization: str | None,
) -> tuple[TabbitClient, str, str]:
    """返回 (client, token_name, token_id)"""
    # 若 token 池非空，走轮询
    if _tm.has_tokens:
        # 校验 proxy api_key
        api_key = _cfg.get("proxy", "api_key")
        if api_key:
            bearer = (authorization or "").replace("Bearer ", "")
            if bearer != api_key:
                raise HTTPException(status_code=401, detail="invalid api key")
        token_info, client = await _tm.get_next()
        if token_info is None:
            raise HTTPException(
                status_code=503, detail="no available tokens (all cooling down)"
            )
        return client, token_info.get("name", "unknown"), token_info["id"]

    # fallback: 从 Authorization 读 token（向后兼容）
    token = (authorization or "").replace("Bearer ", "")
    if not token:
        raise HTTPException(status_code=401, detail="missing token")
    if token not in _fallback_clients:
        _fallback_clients[token] = TabbitClient(
            token,
            _cfg.get("tabbit", "base_url"),
            _cfg.get("tabbit", "client_id"),
            _cfg.get("tabbit", "proxy_url"),
        )
    return _fallback_clients[token], "bearer", ""


# // 将 Tabbit SSE 转换为 OpenAI 流式响应
async def _stream_handler(client, session_id, content, tabbit_model, req_model, completion_id, token_name, token_id):
    start = time.time()
    error_msg = ""
    got_content = False
    try:
        yield (
            f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
        )

        async for event in client.send_message(session_id, content, tabbit_model):
            et, ed = event["event"], event["data"]
            if et == "message_chunk" and "content" in ed:
                got_content = got_content or bool(ed["content"])
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": ed["content"]},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            elif et in ("message_finish", "finish"):
                yield (
                    f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                )

        if not got_content:
            raise Exception("Tabbit stream finished without message content")
        yield "data: [DONE]\n\n"
        if token_id:
            _tm.report_success(token_id)
    except Exception as e:
        error_msg = str(e)
        if token_id:
            _tm.report_error(token_id)
        raise
    finally:
        duration = time.time() - start
        _logs.add(
            LogEntry(
                model=req_model,
                token_name=token_name,
                stream=True,
                status="success" if not error_msg else "error",
                duration=duration,
                error=error_msg,
            )
        )


# // 处理 OpenAI Chat Completions 请求
@router.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest, authorization: str = Header(None)
):
    client, token_name, token_id = await _get_client_and_token(authorization)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    direct_tool_call = _build_direct_tool_call(req)
    if direct_tool_call:
        try:
            arguments = json.loads(direct_tool_call["function"]["arguments"])
            if is_write_file_tool(direct_tool_call["function"]["name"]):
                arguments = await _fill_generated_file_content(client, req, arguments)
            tool_result = execute_tool_call(
                direct_tool_call["function"]["name"], arguments, _cfg
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        if req.stream:
            return StreamingResponse(
                _stream_tool_result_response(req, completion_id, direct_tool_call, tool_result),
                media_type="text/event-stream",
            )
        return _tool_result_response(req, completion_id, direct_tool_call, tool_result)

    try:
        model_map = await client.get_model_map()
        tabbit_model = model_map[req.model.lower()]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"unsupported model: {req.model}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    content = _build_content(req.messages)

    try:
        session_id = await client.create_chat_session()
    except Exception as e:
        if token_id:
            _tm.report_error(token_id)
        _logs.add(
            LogEntry(
                model=req.model,
                token_name=token_name,
                stream=req.stream,
                status="error",
                error=str(e),
            )
        )
        raise HTTPException(status_code=502, detail=str(e))

    if req.stream:
        return StreamingResponse(
            _stream_handler(
                client,
                session_id,
                content,
                tabbit_model,
                req.model,
                completion_id,
                token_name,
                token_id,
            ),
            media_type="text/event-stream",
        )

    # 非流式
    start = time.time()
    full_text = ""
    error_msg = ""
    try:
        async for event in client.send_message(session_id, content, tabbit_model):
            if event["event"] == "message_chunk":
                full_text += event["data"].get("content", "")
        if not full_text:
            raise Exception("Tabbit response finished without message content")
        if token_id:
            _tm.report_success(token_id)
    except Exception as e:
        error_msg = str(e)
        if token_id:
            _tm.report_error(token_id)
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        duration = time.time() - start
        _logs.add(
            LogEntry(
                model=req.model,
                token_name=token_name,
                stream=False,
                status="success" if not error_msg else "error",
                duration=duration,
                error=error_msg,
            )
        )

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": "stop",
            }
        ],
    }


# // 返回当前 Tabbit 模型列表
@router.get("/v1/models")
async def list_models():
    model_map = await fetch_model_map(
        _cfg.get("tabbit", "base_url") if _cfg else None,
        _cfg.get("tabbit", "proxy_url") if _cfg else None,
    )
    model_ids = list(dict.fromkeys(model_map.keys()))
    return {
        "object": "list",
        "data": [
            {"id": k, "object": "model", "owned_by": "tabbit"}
            for k in model_ids
        ],
    }
