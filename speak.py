"""
用 ElevenLabs 念一段带位置标签的稿子，渲染成双耳立体声 WAV。

  export ELEVENLABS_API_KEY=...
  export ELEVENLABS_VOICE_ID=...
  python speak.py "[右耳][whispers] Don't move. [脑后] I'm right behind you. [左耳] And now I'm here." out.wav

调 API 在 eleven.py，渲染在 binaural_render.py。
"""
import asyncio, json, sys
from binaural_render import render_pcm
from eleven import ElevenLabsError, synthesize

if len(sys.argv) < 3:
    sys.exit(__doc__.strip())
text, out = sys.argv[1], sys.argv[2]

try:
    speech = asyncio.run(synthesize(text))
except (ElevenLabsError, ValueError) as e:
    sys.exit(str(e))

if not speech.aligned:
    print("对齐表和稿子字数对不上，这次不用位置标签", file=sys.stderr)

render_pcm(speech.pcm, speech.sample_rate, out, speech.cues)
print("cues:", json.dumps(speech.cues, ensure_ascii=False))
