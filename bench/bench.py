"""性能基准：吞吐、延迟、内存。

    python bench/bench.py
    python bench/bench.py --seconds 120 --preset far

测三件事：
1. 吞吐：一段音频喂进去要多久，折算成实时倍数。
2. 延迟：从「说完最后一个字」到「切出整句」要多久。
3. 内存：底噪窗口和内部缓冲会占多少。
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import hae_vad  # noqa: E402


def make_audio(seconds: float, rate: int = 16000) -> np.ndarray:
    """造一段「说 2 秒停 1 秒」交替的音频，带底噪。"""
    rng = np.random.default_rng(20260101)
    out = []
    total = 0.0
    speaking = True
    while total < seconds:
        dur = 2.0 if speaking else 1.0
        n = int(rate * dur)
        if speaking:
            t = np.arange(n) / rate
            # 130 Hz 基频加两个谐波，听起来接近人声的频谱重心
            wave = (
                0.06 * np.sin(2 * np.pi * 130 * t)
                + 0.03 * np.sin(2 * np.pi * 260 * t)
                + 0.015 * np.sin(2 * np.pi * 390 * t)
            )
            # 加一点包络起伏，别让它在窗里恒等
            wave *= 0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * t) ** 2
            seg = wave + rng.normal(0, 0.002, n)
        else:
            seg = rng.normal(0, 0.002, n)
        out.append(seg.astype(np.float32))
        total += dur
        speaking = not speaking
    return np.concatenate(out)


def bench_throughput(audio: np.ndarray, rate: int, preset: str, block_ms: float) -> dict:
    cfg = hae_vad.config_preset(preset)
    seg = hae_vad.Segmenter(rate=rate, config=cfg)
    block = max(1, int(rate * block_ms / 1000.0))

    pushes = 0
    sentences = 0
    t0 = time.perf_counter()
    for i in range(0, audio.size, block):
        batch = seg.push(audio[i:i + block])
        pushes += 1
        sentences += len(batch)
    tail = seg.flush()
    sentences += len(tail)
    dt = time.perf_counter() - t0

    audio_s = audio.size / float(rate)
    return {
        "preset": preset,
        "block_ms": block_ms,
        "audio_s": audio_s,
        "wall_s": dt,
        "rtf": audio_s / dt if dt > 0 else float("inf"),
        "pushes": pushes,
        "sentences": sentences,
    }


def bench_latency(rate: int = 16000) -> list[float]:
    """量「语音结束 → 句子吐出」的延迟。

    喂 1 秒人声，然后一直喂静音，数多少毫秒后切出句子。
    """
    results = []
    for preset in ("near", "far", "aggressive"):
        cfg = hae_vad.config_preset(preset)
        seg = hae_vad.Segmenter(rate=rate, config=cfg)
        n_speech = int(rate * 1.0)
        t = np.arange(n_speech) / rate
        speech = (0.08 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)

        block = int(rate * 0.03)
        for i in range(0, speech.size, block):
            seg.push(speech[i:i + block])

        silence = np.zeros(block, dtype=np.float32)
        blocks = 0
        cut = False
        while blocks < 200:  # 最多 6 秒
            blocks += 1
            if seg.push(silence):
                cut = True
                break
        results.append(blocks * 30.0 if cut else float("nan"))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="HAE-VAD 性能基准")
    ap.add_argument("--seconds", type=float, default=60.0, help="测试音频长度")
    ap.add_argument("--preset", default="near", help="预设名")
    ap.add_argument("--block-ms", type=float, default=30.0, help="块长")
    ap.add_argument("--rate", type=int, default=16000)
    args = ap.parse_args()

    audio = make_audio(args.seconds, args.rate)
    print(f"音频 {audio.size / args.rate:.1f} 秒，{args.rate} Hz，{args.block_ms:.0f} ms 一块")
    print()

    print("== 吞吐 ==")
    rows = []
    for preset in hae_vad.config_names():
        r = bench_throughput(audio, args.rate, preset, args.block_ms)
        rows.append(r)
        print(
            f"{preset:11s} 用时 {r['wall_s'] * 1000:7.1f} ms  "
            f"实时倍数 {r['rtf']:8.1f}×  切出 {r['sentences']:3d} 句"
        )

    print()
    print("== 切句延迟（语音结束到吐句） ==")
    lat = bench_latency(args.rate)
    for name, ms in zip(("near", "far", "aggressive"), lat):
        print(f"{name:11s} {ms:6.1f} ms")

    print()
    print("== 内存（单实例） ==")
    seg = hae_vad.Segmenter(rate=args.rate)
    cfg = seg.config
    floor_bytes = cfg.floor_size * 8
    print(f"底噪窗口  {cfg.floor_size} 个 double = {floor_bytes / 1024:.1f} KB")
    print(f"预录缓冲  {cfg.preroll_ms:.0f} ms = {args.rate * cfg.preroll_ms / 1000 * 4 / 1024:.1f} KB")
    print("整包体积（不含 numpy）:")
    pkg = os.path.join(ROOT, "hae_vad")
    total = 0
    for fn in sorted(os.listdir(pkg)):
        if fn.endswith(".py"):
            sz = os.path.getsize(os.path.join(pkg, fn))
            total += sz
            print(f"  {fn:16s} {sz / 1024:6.1f} KB")
    print(f"  {'合计':16s} {total / 1024:6.1f} KB")
    print(f"  头文件 hae_vad.h  {os.path.getsize(os.path.join(ROOT, 'include', 'hae_vad.h')) / 1024:6.1f} KB")

    print()
    best = max(rows, key=lambda r: r["rtf"])
    print(f"最快：{best['preset']}  {best['rtf']:.0f}× 实时")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
