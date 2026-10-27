"""Python 版与 C++ 版的对拍。

编译 ``tests/parity_main.cpp``，跑一遍；再用完全相同的输入跑 Python 版，
逐行比对输出。任何一行不一致都会列出，并返回非零退出码。

    python tests/run_parity.py

需要一台装了 MSVC 的 Windows。没有编译器时脚本会跳过，不算失败。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import hae_vad  # noqa: E402


# ---------------------------------------------------------------- 音频生成
class Lcg:
    """和 C++ 侧 Lcg 结构完全一致。"""

    def __init__(self, seed: int) -> None:
        self.s = seed & 0xFFFFFFFF

    def next(self) -> float:
        self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFF
        bits = (self.s >> 8) & 0x00FFFFFF
        return float(bits) / 16777216.0 * 2.0 - 1.0


def gen_audio(rate: int = 16000) -> np.ndarray:
    def ms_to_n(ms: float) -> int:
        return int(rate * ms / 1000.0)

    def silence(ms: float) -> np.ndarray:
        return np.zeros(ms_to_n(ms), dtype=np.float32)

    def tone(ms: float, amp: float, freq: float) -> np.ndarray:
        n = ms_to_n(ms)
        t = np.arange(n, dtype=np.float64) / float(rate)
        return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)

    parts = [
        silence(500.0),
        tone(900.0, 0.08, 220.0),
        silence(1200.0),
        tone(700.0, 0.06, 180.0),
        silence(1100.0),
        tone(1300.0, 0.10, 260.0),
        silence(1200.0),
    ]
    out = np.concatenate(parts).astype(np.float32)

    rng = Lcg(20260101)
    noise = np.fromiter((rng.next() for _ in range(out.size)), dtype=np.float32, count=out.size)
    out = (out + noise * 0.0015).astype(np.float32)
    return out


def fmt(v: float) -> str:
    return f"{float(v):.6f}"


# ---------------------------------------------------------------- Python 侧输出
def python_report() -> list[str]:
    rate = 16000
    audio = gen_audio(rate)
    lines: list[str] = []
    lines.append(f"rate {rate}")
    lines.append(f"samples {audio.size}")
    lines.append(f"rms {fmt(hae_vad.rms(audio))}")
    lines.append(f"zcr {fmt(hae_vad.zero_crossing_rate(audio))}")

    vals = [float(i) * 0.5 for i in range(64)]
    for pct in (20.0, 50.0, 95.0):
        lines.append(f"pct{int(pct)} {fmt(hae_vad.percentile(vals, pct))}")

    presets = {
        "near": hae_vad.NEAR,
        "far": hae_vad.FAR,
        "noisy": hae_vad.NOISY,
        "aggressive": hae_vad.AGGRESSIVE,
    }
    block = 480
    for name, cfg in presets.items():
        clock = [0.0]
        seg = hae_vad.Segmenter(rate=rate, config=cfg, clock=lambda c=clock: c[0])
        got = []
        partial_calls = 0
        partial_nonempty = 0
        last_partial = 0
        for i in range(0, audio.size, block):
            batch = seg.push(audio[i:i + block])
            got.extend(batch)
            clock[0] += 0.03
            part = seg.partial()
            partial_calls += 1
            if part is not None and getattr(part, "size", 0) > 0:
                partial_nonempty += 1
                last_partial = int(part.size)
        got.extend(seg.flush())

        lens = ",".join(str(int(s.size)) for s in got)
        lines.append(f"preset {name} count {len(got)} lens {lens}")
        lines.append(
            f"preset {name} partial calls {partial_calls} "
            f"nonempty {partial_nonempty} last {last_partial}"
        )
        lines.append(f"preset {name} floor {fmt(seg.noise_floor)}")

    whole = audio.astype(np.float32)
    lines.append(f"trim_tail {hae_vad.trim_tail_valley(whole, rate).size}")
    lines.append(f"trim_fast {hae_vad.trim_to_valley(whole, rate).size}")
    return lines


# ---------------------------------------------------------------- C++ 侧
def find_msvc() -> str | None:
    """返回 vcvars64.bat 路径，找不到返回 None。"""
    candidates = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def build_and_run_cpp() -> str | None:
    vcvars = find_msvc()
    if vcvars is None:
        return None

    tmp = tempfile.mkdtemp(prefix="hae_vad_parity_")
    try:
        exe = os.path.join(tmp, "parity.exe")
        src = os.path.join(ROOT, "tests", "parity_main.cpp")
        inc = os.path.join(ROOT, "include")
        bat = os.path.join(tmp, "build.bat")
        with open(bat, "w", encoding="mbcs", newline="\r\n") as f:
            f.write("@echo off\r\n")
            f.write(f'call "{vcvars}" >nul\r\n')
            f.write(
                f'cl /nologo /std:c++17 /EHsc /O2 /MD /utf-8 /DNOMINMAX '
                f'/I "{inc}" /Fe:"{exe}" "{src}"\r\n'
            )
            f.write(f'if errorlevel 1 exit /b 1\r\n')
            f.write(f'"{exe}"\r\n')

        proc = subprocess.run(["cmd", "/c", bat], capture_output=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout.decode("mbcs", "replace"))
            sys.stderr.write(proc.stderr.decode("mbcs", "replace"))
            raise RuntimeError("C++ 对拍程序编译或运行失败")
        return proc.stdout.decode("mbcs", "replace")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- 比对
def main() -> int:
    py_lines = python_report()

    cpp_out = build_and_run_cpp()
    if cpp_out is None:
        print("跳过：没找到 MSVC，无法编译 C++ 版")
        return 0

    raw = cpp_out.replace("\r\n", "\n").split("\n")
    if "#BEGIN#" not in raw:
        print("C++ 输出里找不到 #BEGIN# 标记")
        return 1
    cpp_lines = [ln for ln in raw[raw.index("#BEGIN#") + 1:] if ln.strip()]

    ok = True
    if len(py_lines) != len(cpp_lines):
        print(f"行数不一致：Python {len(py_lines)} 行，C++ {len(cpp_lines)} 行")
        ok = False

    for i in range(max(len(py_lines), len(cpp_lines))):
        a = py_lines[i] if i < len(py_lines) else "<缺行>"
        b = cpp_lines[i] if i < len(cpp_lines) else "<缺行>"
        if a.strip() == b.strip():
            continue
        ok = False
        print(f"第 {i + 1} 行不一致")
        print(f"  Python: {a}")
        print(f"  C++   : {b}")

    if ok:
        print(f"对拍通过：{len(py_lines)} 行逐行一致")
        return 0
    print("对拍失败")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
