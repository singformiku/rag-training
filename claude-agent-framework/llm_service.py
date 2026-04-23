"""
LLM Service
===========
對外的 LLM 接入入口，沿用其他 project 內 ``infrastructure`` 的寫法與命名。

與原 (proprietary) 版本的差異：
- ``settings`` 改從本 repo 的 ``src.config`` 載入（從 .env 讀取）
- ``INFO`` 改用 Python stdlib ``logging``
- 純 OpenAI client (未整合 tracing)
- ``aia_auth`` 仍以原路徑 import —— 請確保該套件已安裝

新增 ``complete_with_tools()`` 供 Agent 的 tool-calling 迴圈使用。
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional

import httpx
from aia_auth import auth
from langchain_core.language_models.base import LanguageModelInput
from langchain_openai import ChatOpenAI
from openai import OpenAI

from src.config import settings

# ---------------------------------------------------------------------------
# Logging — 取代原 ``infrastructure.logging.ekg_logging.INFO``
# ---------------------------------------------------------------------------
logger = logging.getLogger("llm_service")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def INFO(msg: str) -> None:  # noqa: N802 — 保留原命名
    logger.info(msg)


# ---------------------------------------------------------------------------
# OAuth Bearer Token 自動更新
# ---------------------------------------------------------------------------
class AuthenticationProviderWithClientSideTokenRefresh(httpx.Auth):
    def __init__(self):
        """
        Initializes the AuthenticationProviderWithTokenRefresh class.

        Initializes the client_id, client_secret, last_refreshed, and valid_until instance variables.
        """
        # Below properties are applicable to OAUTH only
        self.client_id = settings.LLM_CLIENT_ID
        self.client_secret = settings.LLM_SECRET
        self.last_refreshed = math.floor(time.time())
        self.valid_until = math.floor(time.time()) - 1
        self.token: str = ""
        self.expires_in: int = 0

    def auth_flow(self, request):
        """
        Authenticates a request using Client & Secret (OAuth client_credentials).

        Parameters:
            request: The request object to authenticate.

        Returns:
            The authenticated request object.
        """
        request.headers["Authorization"] = f"Bearer {self.get_bearer_token()}"
        yield request

    def get_bearer_token(self) -> str:
        """
        Returns the bearer token. If the current token has expired, it generates a new one using the client ID and secret.

        Returns:
            str: The generated or existing bearer token.
        """
        if self._is_expired():
            INFO("Generating new token...")
            self.last_refreshed = math.floor(time.time())
            _resp = auth.client_credentials(self.client_id, self.client_secret)
            self.token = _resp.token
            self.expires_in = _resp.expires_in
            self.valid_until = self.last_refreshed + self.expires_in
            INFO(
                f"New token generated. expires in: {self.expires_in}, "
                f"valid until: {self.valid_until}"
            )
        return self.token

    def _is_expired(self) -> bool:
        return time.time() >= self.valid_until


# ---------------------------------------------------------------------------
# LLM Service
# ---------------------------------------------------------------------------
class LLMService:
    _http_client = httpx.Client(
        auth=AuthenticationProviderWithClientSideTokenRefresh(),
        verify=settings.PEM_LOCATION if settings.PEM_LOCATION else False,
    )

    def __init__(self):
        INFO(f"Init LLM service with model: {settings.LLM_MODEL}")
        self._llm_client: Optional[OpenAI] = self.client
        self._chat_client: Optional[ChatOpenAI] = self.chat_model

    # ---------------------------------------------------------------------
    # Public properties
    # ---------------------------------------------------------------------
    @property
    def client(self) -> OpenAI:
        """Return an ``openai.OpenAI`` client configured with a valid token."""
        return OpenAI(
            base_url=settings.LLM_BASE_URL,
            api_key="",  # token 已由 http_client 的 auth 注入
            http_client=self._http_client,
        )  # type: ignore[arg-type]

    @property
    def chat_model(self) -> ChatOpenAI:
        """Return a ``ChatOpenAI`` instance using the current token."""
        return ChatOpenAI(
            base_url=settings.LLM_BASE_URL,
            api_key=auth.client_credentials(settings.LLM_CLIENT_ID, settings.LLM_SECRET).token,
            model=settings.LLM_MODEL,
            http_client=self._http_client,
            temperature=0,
            model_kwargs={"seed": 42},
        )

    # ---------------------------------------------------------------------
    # Completion helpers — 對齊原版 API
    # ---------------------------------------------------------------------
    def complete_openai_client(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Return the result of completion of OpenAI client."""
        completion = self.client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens or settings.LLM_MAX_RESPONSE_TOKENS,
            temperature=0.0,
            top_p=0.1,
        )
        return completion.choices[0].message.content

    def complete_openai_client_with_message(
        self,
        message: List[Dict],
        max_tokens: Optional[int] = None,
    ) -> str:
        """Return the result of completion of OpenAI client."""
        completion = self.client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=message,
            max_tokens=max_tokens or settings.LLM_MAX_RESPONSE_TOKENS,
            temperature=0.0,
            top_p=0.1,
        )
        return completion.choices[0].message.content

    def invoke_chatopenai_client(self, model_input: LanguageModelInput) -> str:
        """Return the result of invocation of ChatOpenAI client."""
        response = self.chat_model.invoke(model_input)
        return response.content

    # ---------------------------------------------------------------------
    # Agent tool-calling helper (新增)
    # ---------------------------------------------------------------------
    def complete_with_tools(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 0.1,
        reasoning_effort: Optional[str] = None,
    ):
        """
        呼叫 chat.completions 並回傳原始 completion 物件 (給 Agent 迴圈解析)。

        - ``tools``：OpenAI function-calling 格式列表，或 None
        - ``tool_choice``："auto" / "none" / {"type":"function","function":{...}}
        - ``reasoning_effort``：gpt-oss 系列的推理強度 (low/medium/high)，或 None

        回傳 ``ChatCompletion`` 物件。呼叫端使用 ``completion.choices[0].message``
        取得訊息 (含 .content 與 .tool_calls) 以及 ``completion.choices[0].finish_reason``。
        """
        kwargs: Dict[str, Any] = {
            "model": settings.LLM_MODEL,
            "messages": messages,
            "max_tokens": max_tokens or settings.LLM_MAX_RESPONSE_TOKENS,
            "temperature": temperature,
            "top_p": top_p,
        }
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
        if reasoning_effort:
            # gpt-oss-120b: low/medium/high；伺服器不支援時會被忽略或 reject
            kwargs["reasoning_effort"] = reasoning_effort

        return self.client.chat.completions.create(**kwargs)


llm_service = LLMService()
