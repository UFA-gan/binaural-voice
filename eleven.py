"""
调 ElevenLabs 念一段带位置标签的稿子，返回 PCM 和换算成秒的 cues。

给 speak.py（命令行）和 mcp_server.py（MCP 工具）共用，只此一处调 API。

位置标签必须在发请求前拿掉：eleven_v4 会把不认识的方括号当演出指令去演。
`[whispers]` 这类演出标签不匹配 TAG，原样透传，是故意保留的。
"""
import base64
import os
import re
from dataclasses import dataclass, field

import httpx

API_BASE = "https://api.elevenlabs.io/v1"
TAG = re.compile(r"\[(左耳|右耳|脑后|面前|贴近|退开)\]")
VOICE_ID_RE = re.compile(r"\A[A-Za-z0-9_-]{1,64}\Z")

DEFAULT_MODEL = "eleven_v4"
DEFAULT_FORMAT = "pcm_24000"   # pcm_44100 只有 Pro 及以上套餐能用
DEFAULT_TIMEOUT = 120.0


class ElevenLabsError(RuntimeError):
    """API 返回了非 2xx。message 里带状态码和响应体前 300 字。"""


@dataclass
class Speech:
    pcm: bytes                       # 16-bit 单声道 PCM
    sample_rate: int
    cues: list[dict] = field(default_factory=list)   # [{"tag": "右耳", "t": 1.23}]
    tts_text: str = ""               # 实际发出去的稿子（标签已剥掉）
    aligned: bool = True             # 对齐表字数对不上时为 False，此时 cues 为空


def split_tags(text):
    """剥掉位置标签，返回 (发出去的稿子, [(标签, 落在剥掉后第几个字)])。"""
    tts, marks, last = "", [], 0
    for m in TAG.finditer(text):
        tts += text[last:m.start()]
        marks.append((m.group(1), len(tts)))
        last = m.end()
    return tts + text[last:], marks


def _sample_rate(output_format):
    if not output_format.startswith("pcm_"):
        raise ValueError(f"output_format 必须是 pcm_*（渲染要的是裸 PCM），收到 {output_format!r}")
    return int(output_format.split("_")[1])


async def synthesize(text, *, api_key=None, voice_id=None, model_id=None, output_format=None, timeout=DEFAULT_TIMEOUT):
    """念 text，返回 Speech。

    api_key / voice_id / model_id / output_format 留空则读环境变量
    ELEVENLABS_API_KEY / ELEVENLABS_VOICE_ID / ELEVENLABS_MODEL_ID / ELEVENLABS_FORMAT。
    """
    api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
    voice_id = voice_id or os.environ.get("ELEVENLABS_VOICE_ID")
    model_id = model_id or os.environ.get("ELEVENLABS_MODEL_ID", DEFAULT_MODEL)
    output_format = output_format or os.environ.get("ELEVENLABS_FORMAT", DEFAULT_FORMAT)
    if not api_key:
        raise ValueError("缺少 ELEVENLABS_API_KEY")
    if not voice_id:
        raise ValueError("缺少 ELEVENLABS_VOICE_ID")
    # voice_id 会拼进 URL 路径，而它可能来自模型/用户。挡住 ../ 和 ? 之类，
    # 否则能把请求引到别的路径上去。
    if not VOICE_ID_RE.match(voice_id):
        raise ValueError(f"voice_id 格式不对：{voice_id!r}")

    fs = _sample_rate(output_format)
    tts_text, marks = split_tags(text)

    # with-timestamps 端点：一次请求同时拿音频和逐字起止时间，价格跟普通接口一样
    url = f"{API_BASE}/text-to-speech/{voice_id}/with-timestamps?output_format={output_format}"
    payload = {"text": tts_text, "model_id": model_id}
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload, headers={"xi-api-key": api_key})
    if r.status_code >= 400:
        raise ElevenLabsError(f"ElevenLabs {r.status_code}: {r.text[:300]}")

    d = r.json()

    # 标签换成秒数：取它后面那个字的开始时间（对齐表和发出去的稿子逐字对应；对不上就不用标签）
    starts = d["alignment"]["character_start_times_seconds"]
    aligned = len(starts) == len(tts_text)
    cues = [{"tag": tag, "t": starts[min(at, len(starts) - 1)]} for tag, at in marks] if aligned else []

    return Speech(
        pcm=base64.b64decode(d["audio_base64"]),
        sample_rate=fs,
        cues=cues,
        tts_text=tts_text,
        aligned=aligned,
    )
