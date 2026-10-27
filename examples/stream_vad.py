"""把一段 wav 流式喂给 Segmenter，模拟实时麦克风的效果。

    python examples/stream_vad.py 你的录音.wav
    python examples/stream_vad.py 你的录音.wav -p far

和 cli.py 的区别：这里按 30 ms 一块慢慢喂，中间打印临时结果，
真实还原「边说边出字」的时序，而不是一次性处理整段文件。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import hae_vad  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="流式切句示例")
    ap.add_argument("wav", help="16 位单声道 wav 文件")
    ap.add_argument("-p", "--preset", default="near", help="预设：near/far/noisy/aggressive")
    ap.add_argument("--realtime", action="store_true", help="按真实时间睡眠，模拟实时输入")
    args = ap.parse_args()

    audio, rate = hae_vad.read_wav(args.wav)
    info = hae_vad.read_info(args.wav)
    print(f"文件 {args.wav}：{info.samples / float(rate):.2f} 秒，{rate} Hz，{info.channels} 声道")
    print(f"预设 {args.preset}")
    print()

    seg = hae_vad.Segmenter(rate=rate, config=hae_vad.config_preset(args.preset))
    block = int(rate * 0.03)

    t0 = time.perf_counter()
    clock = 0.0  # 已处理的音频时间
    last_partial = 0.0
    n_sent = 0

    for i in range(0, audio.size, block):
        chunk = audio[i:i + block]
        sentences = seg.push(chunk)
        clock += chunk.size / float(rate)

        # 临时结果按库里的节流间隔触发，这里只负责在控制台原地刷新
        part = seg.partial()   # 未到刷新间隔时返回 None
        if part is not None and part.size > 0 and clock - last_partial >= 0.5:
            last_partial = clock
            bar = "#" * int(min(40, part.size / float(rate) * 12))
            sys.stdout.write(f"\r  [{clock:6.2f}s] 正在说：{bar:<40s}")
            sys.stdout.flush()

        for s in sentences:
            n_sent += 1
            dur = s.size / float(rate)
            sys.stdout.write("\r" + " " * 70 + "\r")
            print(f"  第 {n_sent} 句  {dur:5.2f} 秒  ({s.size} 样本)")

        if args.realtime:
            time.sleep(chunk.size / float(rate))

    for s in seg.flush():
        n_sent += 1
        print(f"  第 {n_sent} 句（收尾）  {s.size / float(rate):5.2f} 秒  ({s.size} 样本)")

    dt = time.perf_counter() - t0
    total = audio.size / float(rate)
    print()
    print(f"共 {n_sent} 句，音频 {total:.2f} 秒，处理用时 {dt * 1000:.1f} ms"
          f"（{total / dt:.0f}× 实时）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
