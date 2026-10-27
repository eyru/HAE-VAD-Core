"""片段尾部裁切。

切句时为了保住字尾，片段里会带着 hang_ms（默认 700 ms）的尾部静音。
静音太长喂给 ASR 有两个坏处：一是浪费算力，二是模型容易在长静音里「编」
出没说过的话。

所以从后往前找第一个真正的静音点，切在那里，再留一小段安全余量。
"""

from __future__ import annotations

import numpy as np

__all__ = ["trim_tail_valley", "trim_to_valley", "energy_envelope"]

_EPS = 1e-12


def energy_envelope(seg, rate=16000, win_ms=10.0):
    """按 win_ms 一窗算能量包络，返回 (energies, step)。"""
    a = np.asarray(seg, dtype=np.float64).ravel()
    step = max(1, int(rate * float(win_ms) / 1000.0))
    n = a.size
    if n < step:
        return np.zeros(0, dtype=np.float64), step
    count = (n - step) // step + 1
    out = np.empty(count, dtype=np.float64)
    for i in range(count):
        chunk = a[i * step:(i + 1) * step]
        out[i] = float(np.mean(chunk * chunk))
    return out, step


def trim_tail_valley(seg, rate=16000, search_ms=320.0, ratio=0.02, pad_ms=40.0,
                     min_ms=40.0, win_ms=10.0):
    """把片段尾部多余的静音切到能量谷底。

    从尾部往前 search_ms 的范围里找第一个能量低于峰值 ratio 的窗口，
    切在那里，再往后留 pad_ms 的安全余量。

    峰值取**整段**而不是搜索窗口内的最大值。这一点很容易搞错：
    如果尾部静音比 search_ms 还长，窗口里全是静音，窗口内峰值接近零，
    任何阈值判断都会失效，整段原样返回，裁切等于没做。

    找不到谷底、片段太短、能量全零，都原样返回——裁切失败不该丢音频。
    """
    a = np.asarray(seg, dtype=np.float32).ravel()
    n = a.size
    if n == 0:
        return a
    step = max(1, int(rate * float(win_ms) / 1000.0))
    if n <= step * 4:
        return a
    if n < rate * float(min_ms) / 1000.0:
        return a

    env, _ = energy_envelope(a, rate=rate, win_ms=win_ms)
    if env.size == 0:
        return a
    peak = float(np.max(env))
    if peak <= _EPS:
        return a

    # 从尾部往前跳过安静窗口，停在第一个「还在说话」的窗口上，切在它之后。
    #
    # 两个坑都在这一小段里：
    # 1. 不能写成「找到第一个安静窗口就切在那里」——从尾部数第一个窗口本来
    #    就是安静的，那样会切在倒数第一个窗口，等于什么都没裁。
    # 2. 搜索范围不能只有 search_ms。尾部静音常常比 search_ms 长（hang_ms
    #    默认 700 ms），只搜最后 320 ms 会全部落在静音里，一个响亮窗口都
    #    找不到，结果还是原样返回。
    cut = n
    for i in range(env.size - 1, -1, -1):
        if env[i] >= peak * float(ratio):
            cut = (i + 1) * step
            break

    cut = min(n, cut + int(rate * float(pad_ms) / 1000.0))
    if cut <= step * 4:
        return a
    return a[:cut].copy()


def trim_to_valley(seg, rate=16000, search_ms=320.0):
    """流式临时结果用的轻量裁切。

    临时结果只影响「边说边出字」的显示，不需要精确，所以只按平均能量
    从尾部剥掉明显安静的部分，不做逐窗搜索。
    """
    a = np.asarray(seg, dtype=np.float32).ravel()
    n = a.size
    if n == 0:
        return a
    step = max(1, int(rate * 0.01))
    search = max(step, int(rate * float(search_ms) / 1000.0))
    start = max(0, n - search)
    if start >= n:
        return a

    tail = a[start:]
    mean = float(np.mean(tail.astype(np.float64) ** 2))
    if mean <= _EPS:
        return a

    # 从尾部往回剥：连续安静窗口直接丢掉，遇到第一个超过均值的窗口就停
    cut = n
    for i in range(n - step, start, -step):
        chunk = a[i:i + step].astype(np.float64)
        if float(np.mean(chunk * chunk)) >= mean:
            break
        cut = i
    if cut <= step * 4:
        return a
    return a[:cut].copy()
