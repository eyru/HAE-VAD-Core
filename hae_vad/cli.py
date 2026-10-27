"""命令行：把一个 wav 切成句子，打印时间轴，可选把每句导成单独文件。

    hae-vad meeting.wav
    hae-vad meeting.wav -p far
    hae-vad meeting.wav --export out_dir
    hae-vad --print-presets
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from . import Segmenter, __version__
from .config import get as get_preset
from .config import names as preset_names
from .wav import read_wav, write_wav


def _fmt_time(seconds):
    m = int(seconds // 60)
    s = seconds - m * 60
    return "%d:%06.3f" % (m, s)


def run_file(path, preset="near", export_dir=None, block_ms=30.0, scale=1.0):
    audio, rate = read_wav(path)
    cfg = get_preset(preset)
    seg = Segmenter(rate=rate, config=cfg, scale=scale)

    block = max(1, int(rate * block_ms / 1000.0))
    sentences = []
    cursor = 0
    while cursor < audio.size:
        piece = audio[cursor:cursor + block]
        cursor += block
        for s in seg.push(piece):
            sentences.append(s)
    sentences.extend(seg.flush())

    total_s = audio.size / float(rate)
    speech_s = sum(s.size for s in sentences) / float(rate)

    print("文件    %s" % path)
    print("       采样率 %d Hz，%.2f 秒" % (rate, total_s))
    print("预设    %s（门限下限 %.4f，静音 %.0f ms 切句）"
          % (preset, cfg.min_rms, cfg.hang_ms))
    print("结果    %d 句，语音合计 %.2f 秒（占 %.1f%%）"
          % (len(sentences), speech_s, 100.0 * speech_s / total_s if total_s else 0.0))
    print()

    if not sentences:
        print("没有切出句子。可能整段都是静音，或者门限太高——试试 -p far 或 -p aggressive。")
        return 0

    pos = 0
    for i, s in enumerate(sentences):
        # 片段在原始音频里的位置：从头累加
        start_s = pos / float(rate)
        pos += s.size
        dur = s.size / float(rate)
        print("  %3d  %s → %s  (%.2f 秒, %d 样本)"
              % (i + 1, _fmt_time(start_s), _fmt_time(start_s + dur), dur, s.size))
        if export_dir:
            os.makedirs(export_dir, exist_ok=True)
            out = os.path.join(export_dir, "seg_%03d.wav" % (i + 1))
            write_wav(out, s, rate)

    if export_dir:
        print()
        print("已导出到 %s" % os.path.abspath(export_dir))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="hae-vad",
        description="流式语音活动检测与切句")
    ap.add_argument("input", nargs="?", help="输入 WAV")
    ap.add_argument("-p", "--preset", default="near",
                    help="预设：%s" % ", ".join(preset_names()))
    ap.add_argument("--scale", type=float, default=1.0,
                    help="门限场景系数，默认 1.0；0.65 约等于远场")
    ap.add_argument("--block-ms", type=float, default=30.0,
                    help="按多长的块喂进去，默认 30 ms（和实时采集一致）")
    ap.add_argument("--export", metavar="DIR", help="把每句导出成单独 wav")
    ap.add_argument("--print-presets", action="store_true", help="列出所有预设")
    ap.add_argument("--version", action="store_true", help="打印版本")
    args = ap.parse_args(argv)

    if args.version:
        print("hae-vad %s" % __version__)
        return 0

    if args.print_presets:
        for name in preset_names():
            cfg = get_preset(name)
            print("%-12s min_rms=%.4f  hang=%4.0fms  max=%5.0fms  zcr_max=%.2f  factor=%.1f"
                  % (name, cfg.min_rms, cfg.hang_ms, cfg.max_speech_ms,
                     cfg.zcr_max, cfg.floor_factor))
        return 0

    if not args.input:
        ap.print_help()
        return 2

    if not os.path.exists(args.input):
        print("找不到文件：%s" % args.input, file=sys.stderr)
        return 2

    return run_file(args.input, preset=args.preset, export_dir=args.export,
                    block_ms=args.block_ms, scale=args.scale)


if __name__ == "__main__":
    sys.exit(main())
