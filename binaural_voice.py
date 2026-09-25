"""
单声道人声 → 双耳立体声（KU100 近场 HRIR），声音在听者头边移动。只对耳机成立。

  python binaural_voice.py in.wav out.wav                  # 没有位置标签：随机抽一种走法
  python binaural_voice.py in.wav out.wav '[{"t": 0, "tag": "右耳"}, {"t": 2.4, "tag": "脑后"}]'

t   = 从第几秒起去那儿（秒）
tag = 左耳 / 右耳 / 脑后 / 面前（去那儿）· 贴近 / 退开（只改远近）

依赖 numpy、scipy。只读写 WAV；mp3 先转：ffmpeg -i in.mp3 -ac 1 in.wav

渲染逻辑在 binaural_render.py，这里只是个命令行外壳。
"""
import json, sys
from binaural_render import render

if len(sys.argv) < 3:
    sys.exit(__doc__.strip())

src, dst = sys.argv[1], sys.argv[2]
tag_cues = json.loads(sys.argv[3]) if len(sys.argv) > 3 else []
render(src, dst, tag_cues)
