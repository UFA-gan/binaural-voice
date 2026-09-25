"""
binaural-voice 的 remote MCP 服务，给 ChatGPT 之类的客户端用。

  export ELEVENLABS_API_KEY=...
  export ELEVENLABS_VOICE_ID=...
  export MCP_SECRET=$(python -c "import secrets;print(secrets.token_hex(16))")
  uvicorn mcp_server:app --host 0.0.0.0 --port 8080

客户端连 `<base>/<MCP_SECRET>/mcp`。渲染好的 WAV 由 `<base>/<MCP_SECRET>/audio/<id>.wav` 托管。

两个设计上的硬约束，改动前先读：
1. 音频只能靠 URL 回去。MCP 协议有 base64 的 AudioContent，但 OpenAI 的 tool result 只接受
   text part，audio 会被静默丢掉。所以工具返回的是链接，不是音频本身。
2. ChatGPT 的 connector 表单只有 OAuth / No Authentication，没地方填 bearer token。所以门禁
   是 URL 里的 MCP_SECRET，配套的字数上限和速率限制才是真正的止损手段。
"""
import asyncio
import json
import logging
import os
import re
import tempfile
import time
import uuid
from contextvars import ContextVar
from pathlib import Path

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response

from mcp.server import MCPServer

from binaural_render import render_pcm
from eleven import ElevenLabsError, synthesize

log = logging.getLogger("binaural-voice")

MCP_SECRET = os.environ.get("MCP_SECRET", "").strip()
if not MCP_SECRET:
    raise SystemExit(
        "缺少 MCP_SECRET。它同时是 MCP 端点和音频链接的路径前缀，是唯一的访问门禁：\n"
        '  export MCP_SECRET=$(python -c "import secrets;print(secrets.token_hex(16))")\n'
        "本地随便跑一个的话：export MCP_SECRET=devsecret"
    )

MAX_CHARS = int(os.environ.get("MAX_CHARS", "1500"))
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "6"))
AUDIO_TTL_SECONDS = int(os.environ.get("AUDIO_TTL_SECONDS", "3600"))
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
ALLOWED_HOSTS = [h.strip() for h in os.environ.get("ALLOWED_HOSTS", "").split(",") if h.strip()]
AUDIO_DIR = Path(os.environ.get("AUDIO_DIR") or Path(tempfile.gettempdir()) / "binaural-audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# 工具签名里拿不到 Request（MCP 的 custom_route 拿得到，tool 不行），所以用中间件把
# Host 和客户端 IP 塞进 ContextVar。ContextVar 会随 asyncio 任务树向下复制，工具能读到。
_CUR_HOST: ContextVar[str | None] = ContextVar("cur_host", default=None)
_CUR_IP: ContextVar[str | None] = ContextVar("cur_ip", default=None)


class _RequestContextMiddleware:
    """纯 ASGI 中间件。uvicorn 接受任意 ASGI callable，所以可以直接套在 MCP app 外面。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or []}
        host = headers.get("x-forwarded-host") or headers.get("host")
        ip = (headers.get("x-forwarded-for") or "").split(",")[0].strip()
        if not ip:
            client = scope.get("client")
            ip = client[0] if client else "unknown"
        t_host, t_ip = _CUR_HOST.set(host), _CUR_IP.set(ip)
        try:
            await self.app(scope, receive, send)
        finally:
            _CUR_HOST.reset(t_host)
            _CUR_IP.reset(t_ip)


def _base_url() -> str | None:
    """音频链接的前缀。拿不到就返回 None，让调用方回一条能看懂的配置错误。"""
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    host = _CUR_HOST.get()
    if not host:
        return None
    scheme = "http" if host.split(":")[0] in ("127.0.0.1", "localhost", "[::1]") else "https"
    return f"{scheme}://{host}"


# ── 速率限制：滑动窗口，全局 + 按 IP 各算一份 ──
_WINDOW = 60.0
_hits: dict[str, list[float]] = {}


def _allow(key: str, limit: int) -> bool:
    if limit <= 0:
        return True
    now = time.monotonic()
    if len(_hits) > 2000:          # 别让字典无限长
        for k in [k for k, v in _hits.items() if not v or now - v[-1] > _WINDOW]:
            _hits.pop(k, None)
    q = _hits.setdefault(key, [])
    while q and now - q[0] > _WINDOW:
        q.pop(0)
    if len(q) >= limit:
        return False
    q.append(now)
    return True


def _sweep():
    """删掉过期音频。每次请求顺手做，不引额外的定时器依赖。"""
    cutoff = time.time() - AUDIO_TTL_SECONDS
    try:
        for p in AUDIO_DIR.iterdir():
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
    except OSError:
        log.warning("清理 %s 失败", AUDIO_DIR, exc_info=True)


mcp = MCPServer(
    "binaural-voice",
    instructions=(
        "把稿子念成贴在耳边、绕着头走的双耳立体声。只对耳机成立。"
        "调用 speak 时请在稿子里写位置标签（[左耳] [右耳] [脑后] [面前] [贴近] [退开]）来控制声音的位置。"
    ),
)


@mcp.tool()
async def speak(text: str, voice_id: str | None = None) -> str:
    """用 ElevenLabs 念稿子，渲染成贴在耳边、绕着头走的双耳立体声。只对耳机成立。

    在稿子里写位置标签来控制声音的位置，中英文稿子都行：

      [左耳] 到左耳，25cm        [右耳] 到右耳，25cm
      [脑后] 到脑后，约 31cm
      [面前] 到面前，约 42cm
      [贴近] 只改远近：贴回 25cm
      [退开] 只改远近：退到 50cm，声音会自然变远变轻

    写在开头就是一开始就在那个位置；写在句子中间就是从那里开始一边说一边挪过去。
    挪动只在出声时推进，换气停顿时停在原地，所以不会从一只耳朵瞬移到另一只。
    一个标签都没写的话随机抽一种走法。

    其他方括号内容不是位置标签，会原样透传给模型当演出指令 —— [whispers]、[laughs]
    这类可以放心用，很适合营造气氛。

    返回一个 URL，指向渲染好的 24kHz 立体声 WAV。把链接给用户，并提醒戴耳机。
    一次别写太长（1500 字符以内），长稿子分成几次调用。

    Args:
        text: 带位置标签的稿子。
        voice_id: 想换音色时传 ElevenLabs 的 voice id，留空用服务端默认音色。
    """
    text = (text or "").strip()
    if not text:
        return "稿子是空的，没东西可念。"
    if len(text) > MAX_CHARS:
        return (
            f"稿子 {len(text)} 字符，超过上限 {MAX_CHARS}。"
            f"请拆成几段，每次调用一段。"
        )

    if not _allow("global", RATE_LIMIT_PER_MIN) or not _allow(_CUR_IP.get() or "unknown", RATE_LIMIT_PER_MIN):
        return f"请求太频繁了，每分钟最多 {RATE_LIMIT_PER_MIN} 次。等一会儿再试。"

    base = _base_url()
    if not base:
        return "服务端没配好：拿不到对外域名，请设置 PUBLIC_BASE_URL 环境变量。"

    try:
        speech = await synthesize(text, voice_id=voice_id)
    except ElevenLabsError as e:
        return f"ElevenLabs 报错：{e}"
    except ValueError as e:
        return f"服务端配置有问题：{e}"

    if not speech.aligned:
        log.warning("对齐表字数对不上，这次退化成随机走位")

    name = f"{uuid.uuid4().hex}.wav"
    try:
        dur, cues = await asyncio.to_thread(
            render_pcm, speech.pcm, speech.sample_rate, str(AUDIO_DIR / name), speech.cues
        )
    except Exception as e:                      # 渲染失败不该把 MCP 连接搞崩
        log.exception("渲染失败")
        return f"渲染失败：{type(e).__name__}: {e}"

    _sweep()
    url = f"{base}/{MCP_SECRET}/audio/{name}"
    log.info("渲染完成 %.1fs -> %s", dur, name)

    walk = "（稿子里没有位置标签，这次是随机走位）" if not speech.cues else json.dumps(cues, ensure_ascii=False)
    return (
        f"已渲染 {dur:.1f} 秒双耳立体声（24kHz）。\n"
        f"{url}\n"
        f"走位：{walk}\n"
        f"把链接给用户，并提醒戴耳机 —— 双耳效果在音箱上不存在。"
    )


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(request: Request) -> Response:
    return JSONResponse({"status": "ok"})


@mcp.custom_route("/", methods=["GET"])
async def root(request: Request) -> Response:
    """只是让人在浏览器里确认服务活着。不泄露 secret。"""
    return JSONResponse(
        {
            "service": "binaural-voice",
            "mcp_endpoint": "/<MCP_SECRET>/mcp",
            "health": "/healthz",
        }
    )


_AUDIO_NAME = re.compile(r"\A[0-9a-f]{32}\.wav\Z")


@mcp.custom_route(f"/{MCP_SECRET}/audio/{{name}}", methods=["GET"])
async def audio(request: Request) -> Response:
    """托管渲染结果。文件名是 uuid4（122 bit 熵）已经是能力式 URL，secret 前缀算纵深防御。"""
    name = request.path_params["name"]
    if not _AUDIO_NAME.match(name):             # 正则只放行 32 位 hex，顺带挡掉路径穿越
        return Response("not found", status_code=404)
    path = AUDIO_DIR / name
    if not path.is_file():
        return Response("not found", status_code=404)
    _sweep()
    return FileResponse(path, media_type="audio/wav")


def _build_app():
    kwargs = dict(
        streamable_http_path=f"/{MCP_SECRET}/mcp",
        stateless_http=True,     # 不存 session，方便横向扩容
        json_response=True,      # 不走 SSE，纯 JSON 响应，过网关/CDN 更稳
    )
    if ALLOWED_HOSTS:
        # 开了 DNS rebinding 保护。注意：一旦显式传 transport_security，下面的 host 参数
        # 就不再触发自动配置，所以这里不需要也不能靠 host 来关保护。
        from mcp.server.transport_security import TransportSecuritySettings

        kwargs["transport_security"] = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=ALLOWED_HOSTS,
            allowed_origins=[f"https://{h}" for h in ALLOWED_HOSTS],
        )
    else:
        # 必须显式传 0.0.0.0：host 默认是 "127.0.0.1"，MCP SDK 见到它就会自动开启 DNS
        # rebinding 保护并只放行 localhost 的 Host 头，线上请求会全部被拒（421）。
        # 本地测不出来，只有部署后才炸。
        kwargs["host"] = "0.0.0.0"
    return _RequestContextMiddleware(mcp.streamable_http_app(**kwargs))


app = _build_app()
