FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# 依赖单独一层，改代码不用重装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 非 root 运行。渲染结果写在 /tmp 下（世界可写），监听 8080（>1024），都不需要额外授权。
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

# 不需要 ffmpeg：ElevenLabs 回来的是裸 PCM，直接由标准库 wave 写 WAV，没有解码环节。
EXPOSE 8080
CMD ["sh", "-c", "uvicorn mcp_server:app --host 0.0.0.0 --port ${PORT:-8080}"]
