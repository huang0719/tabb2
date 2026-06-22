import re
import json
import uuid
import hashlib
import base64
import hmac
import time
import secrets
import urllib.parse
from typing import AsyncGenerator

import httpx

SIGN_KEY = "f8d0e6a73f8d4b1a9c3d2e1f9a4b7c6d"
TABBIT_VERSION_CONTEXT = "0.33.13(10033013)"
MODEL_MAP = {}


# // 将 Tabbit 展示模型名转换为 OpenAI 风格模型 id
def normalize_model_id(display_name: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", display_name.lower()).strip("-")


# // 从 Tabbit 当前模型配置接口读取模型映射
async def fetch_model_map(
    base_url: str | None = None, proxy_url: str | None = None
) -> dict[str, str]:
    api_base = base_url or "https://web.tabbitbrowser.com"
    async with httpx.AsyncClient(timeout=15, verify=False, proxy=proxy_url or None) as client:
        resp = await client.get(
            f"{api_base}/proxy/v1/model_config/models",
            params={"a": "0", "scene": "chat"},
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
                "Accept": "application/json",
                "Referer": f"{api_base}/chat/new",
            },
        )
    if resp.status_code != 200:
        raise Exception(f"Tabbit model API error {resp.status_code}: {resp.text}")

    body = resp.json()
    models = body.get("models")
    if not isinstance(models, list):
        raise Exception("Tabbit model API response missing models")

    model_map: dict[str, str] = {}
    for item in models:
        display_name = item.get("display_name") if isinstance(item, dict) else None
        if not display_name:
            raise Exception(f"Tabbit model item missing display_name: {item}")
        model_id = normalize_model_id(display_name)
        model_map[model_id] = display_name
        model_map[display_name.lower()] = display_name

    if "default" in model_map:
        model_map["best"] = model_map["default"]
    return model_map


# // 生成 Tabbit 前端 unique-uuid
def make_unique_uuid(is_default_browser: bool = True) -> str:
    hex_chars = "0123456789abcdef"
    fallback_chars = hex_chars.replace("1", "")
    marker_pos = 5
    timestamp_positions = [2, 7, 11, 14, 18, 21, 25, 28]
    timestamp = hex(int(time.time()))[2:].rjust(len(timestamp_positions), "0")[
        -len(timestamp_positions) :
    ]
    timestamp_map = dict(zip(timestamp_positions, timestamp))

    value = ""
    for index in range(32):
        if index == marker_pos:
            value += "1" if is_default_browser else secrets.choice(fallback_chars)
        elif index in timestamp_map:
            value += timestamp_map[index]
        else:
            value += secrets.choice(hex_chars)
    return "-".join(
        [value[:8], value[8:12], value[12:16], value[16:20], value[20:]]
    )


# // 操作 Tabbit Web API
class TabbitClient:
    # // 初始化 Tabbit 客户端和认证字段
    def __init__(
        self,
        token_str: str,
        base_url: str | None = None,
        client_id: str | None = None,
        proxy_url: str | None = None,
    ):
        parts = token_str.split("|")
        self.jwt_token = parts[0]
        self.next_auth = parts[1] if len(parts) > 1 else None
        self.device_id = parts[2] if len(parts) > 2 else str(uuid.uuid4())
        self.user_id = self._extract_user_id(self.jwt_token)
        self.base_url = base_url or "https://web.tabbitbrowser.com"
        self.client_id = client_id or "e7fa44387b1238ef1f6f"
        self.proxy_url = proxy_url or None

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15, read=120, write=15, pool=15),
            follow_redirects=False,
            verify=False,
            proxy=self.proxy_url,
        )

    # // 从 JWT 中提取用户标识
    def _extract_user_id(self, token: str) -> str:
        try:
            payload = json.loads(
                base64.urlsafe_b64decode(token.split(".")[1] + "==")
            )
            return payload.get("id", payload.get("sub", str(uuid.uuid4())))
        except Exception:
            return str(uuid.uuid4())

    # // 生成 Tabbit 请求头
    def _get_headers(self, referer_path: str = "/newtab") -> dict:
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
            "sec-ch-ua": '"Not:A-Brand";v="99", "Tabbit";v="145", "Chromium";v="145"',
            "sec-ch-ua-platform": '"Windows"',
            "x-req-ctx": base64.b64encode(
                TABBIT_VERSION_CONTEXT.encode("utf-8")
            ).decode("ascii"),
            "x-next-intl-locale": "zh-CN",
            "x-chrome-id-consistency-request": (
                f"version=1,client_id={self.client_id},"
                f"device_id={self.device_id},sync_account_id={self.user_id},"
                "signin_mode=all_accounts,signout_mode=show_confirmation"
            ),
            "referer": f"{self.base_url}{referer_path}",
        }

    # // 生成 Tabbit 登录 Cookie
    def _get_cookies(self) -> dict:
        cookies = {
            "token": self.jwt_token,
            "user_id": self.user_id,
            "managed": "tab_browser",
            "NEXT_LOCALE": "zh",
        }
        if self.next_auth:
            cookies["next-auth.session-token"] = self.next_auth
        return cookies

    # // 按当前前端规则生成请求签名
    def _get_sign_headers(self, body_text: str) -> dict:
        timestamp = str(int(time.time() * 1000))
        nonce = secrets.token_hex(16)
        body_hash = hashlib.sha256(body_text.encode("utf-8")).hexdigest()
        sign_text = f"{timestamp}.{nonce}.{body_hash}"
        signature = hmac.new(
            SIGN_KEY.encode("utf-8"),
            sign_text.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {
            "x-timestamp": timestamp,
            "x-signature": nonce,
            "x-nonce": signature,
        }

    # // 创建新的 Tabbit 聊天会话
    async def create_chat_session(self) -> str:
        router_state = [
            "",
            {
                "children": [
                    "chat",
                    {
                        "children": [
                            ["id", "new", "d"],
                            {"children": ["__PAGE__", {}, None, "refetch"]},
                            None,
                            None,
                        ]
                    },
                    None,
                    None,
                ]
            },
            None,
            None,
        ]
        headers = {
            **self._get_headers("/chat/new"),
            "rsc": "1",
            "next-router-state-tree": urllib.parse.quote(json.dumps(router_state)),
        }

        resp = await self.client.get(
            f"{self.base_url}/chat/new",
            params={"_rsc": "auto"},
            headers=headers,
            cookies=self._get_cookies(),
        )

        text = resp.text
        match = re.search(
            r"/chat/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
            text,
        )
        if match:
            return match.group(1)
        raise Exception("Failed to extract chat session_id from RSC response")

    # // 获取当前 Tabbit 支持的模型映射
    async def get_model_map(self) -> dict[str, str]:
        return await fetch_model_map(self.base_url, self.proxy_url)

    # // 向 Tabbit 发送聊天消息并返回 SSE 事件
    async def send_message(
        self, session_id: str, content: str, model: str
    ) -> AsyncGenerator[dict, None]:
        payload = {
            "chat_session_id": session_id,
            "message_id": str(uuid.uuid4()),
            "content": content,
            "selected_model": model,
            "parallel_group_id": None,
            "task_name": "chat",
            "agent_mode": False,
            "metadatas": {"html_content": f"<p>{content}</p>"},
            "references": [],
            "entity": {
                "key": hashlib.md5(b"").hexdigest(),
                "extras": {"type": "tab", "url": ""},
            },
        }
        body_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

        headers = {
            **self._get_headers(f"/chat/{session_id}"),
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "unique-uuid": make_unique_uuid(),
            **self._get_sign_headers(body_text),
        }

        async with self.client.stream(
            "POST",
            f"{self.base_url}/api/v1/chat/completion",
            content=body_text,
            headers=headers,
            cookies=self._get_cookies(),
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise Exception(
                    f"Tabbit API error {resp.status_code}: {body.decode()}"
                )

            current_event = None
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    current_event = line[len("event:") :].strip()
                elif line.startswith("data:") and current_event:
                    data_str = line[len("data:") :].strip()
                    try:
                        data = json.loads(data_str)
                        if current_event == "error":
                            raise Exception(f"Tabbit SSE error: {data}")
                        yield {"event": current_event, "data": data}
                    except Exception as e:
                        if current_event == "error":
                            raise
                        raise Exception(
                            f"Failed to parse Tabbit SSE event {current_event}: {data_str}"
                        ) from e
