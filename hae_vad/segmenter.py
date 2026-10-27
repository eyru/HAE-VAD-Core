"""流式切句状态机。

输入是一块一块的音频，输出是完整的句子。核心状态只有三个：

    active      现在是否在一句话中间
    buf         这句话到目前为止的音频
    silence_ms  尾部已经积累了多少毫秒静音

每收到一块音频做三件事：判断这块是不是人声、按判断更新状态、看要不要切句。

为什么需要预卷（preroll）：/s/ /f/ /θ/ 这类词首的能量比元音低一个数量级，
如果门限只按「当前块」判断，词首会被当成静音丢掉。所以静音时也滚动缓存
一段音频，真正开口时接在前面。这一条直接决定了「Spanish」会不会被听成「fan」。
"""

from __future__ import annotations

import math
import time

import numpy as np

from .config import VadConfig, DEFAULT
from .noise import NoiseFloor, rms
from .trim import trim_tail_valley, trim_to_valley

__all__ = ["Segmenter", "State"]

_EPS = 1e-12


class State:
    """一次切句的中间状态快照，方便调试和测试断言。"""

    __slots__ = ("active", "speech_ms", "silence_ms", "buffered_samples",
                 "level", "threshold", "zcr", "voiced", "segments")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __repr__(self):
        return ("State(active=%s, speech=%.0fms, silence=%.0fms, "
                "level=%.5f, thr=%.5f, zcr=%.3f, voiced=%s, segments=%d)"
                % (self.active, self.speech_ms, self.silence_ms,
                   self.level, self.threshold, self.zcr, self.voiced,
                   self.segments))


def _as_float_block(block):
    """把输入整理成一维 float32；拿不到有效数据就返回 None。"""
    if block is None:
        return None
    try:
        a = np.asarray(block, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if a.ndim > 1:
        # 多声道取第一声道，和采集侧保持一致
        a = a[:, 0] if a.shape[1] > 0 else a.ravel()
    a = a.ravel()
    if a.size == 0:
        return None
    if not np.all(np.isfinite(a)):
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def zero_crossing_rate(block):
    """过零率：相邻样本符号变化的比例，范围 0..1。

    人声在 16 kHz 下大约 0.02–0.25；嘶声、风噪、键盘声普遍 0.4 以上。
    """
    a = np.asarray(block, dtype=np.float64).ravel()
    if a.size < 2:
        return 0.0
    signs = np.signbit(a)
    return float(np.mean(signs[1:] != signs[:-1]))


class Segmenter:
    """流式切句器。

        seg = Segmenter(rate=16000, config=VAAD_FAR)
        for block in mic_blocks:
            for sentence in seg.push(block):
                asr(sentence)

    也可以一次喂整段音频再 flush：

        seg = Segmenter()
        sentences = seg.push(audio) + seg.flush()
    """

    def __init__(self, rate=16000, config=None, scale=1.0, clock=None):
        self.rate = int(rate)
        self._clock = clock or time.time
        self.config = config if config is not None else DEFAULT
        if not isinstance(self.config, VadConfig):
            raise TypeError("config 必须是 VadConfig")
        self._scale = float(scale)

        cfg = self.config
        self._floor = NoiseFloor(size=cfg.floor_size, pct=cfg.floor_percentile,
                                 factor=cfg.floor_factor, min_rms=cfg.min_rms)
        self._pre = []              # 预卷缓存（块列表）
        self._pre_samples = 0
        self._tail = []             # 流式窗口缓存（块列表）
        self._tail_samples = 0
        self._buf = []
        self._active = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._last_partial = -1e9
        self._level = 0.0
        self._threshold = cfg.min_rms
        self._zcr = 0.0
        self._voiced = False
        self._segments = 0

    # ---------------------------------------------------------------- 参数
    def set_scale(self, scale):
        """场景系数：门限整体乘这个值。远场 0.65，强制放大 0.45。"""
        self._scale = float(scale)

    def set_config(self, config):
        """换参数。已积累的音频保留，门限估计不重置。"""
        if not isinstance(config, VadConfig):
            raise TypeError("config 必须是 VadConfig")
        self.config = config
        self._floor = NoiseFloor(size=config.floor_size, pct=config.floor_percentile,
                                 factor=config.floor_factor, min_rms=config.min_rms)

    # ---------------------------------------------------------------- 状态
    @property
    def active(self):
        return self._active

    @property
    def level(self):
        """最近一块的 RMS。"""
        return self._level

    @property
    def threshold(self):
        """当前生效的门限。"""
        return self._threshold

    @property
    def noise_floor(self):
        """当前估计的环境底噪。"""
        return self._floor.floor()

    @property
    def speech_ms(self):
        return self._speech_ms

    def state(self):
        return State(active=self._active, speech_ms=self._speech_ms,
                     silence_ms=self._silence_ms,
                     buffered_samples=sum(b.size for b in self._buf),
                     level=self._level, threshold=self._threshold,
                     zcr=self._zcr, voiced=self._voiced,
                     segments=self._segments)

    def reset(self):
        """清空全部状态。换设备或重启会话时调用。"""
        self._floor.reset()
        self._pre = []
        self._pre_samples = 0
        self._tail = []
        self._tail_samples = 0
        self._buf = []
        self._active = False
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._last_partial = -1e9
        self._level = 0.0
        self._zcr = 0.0
        self._voiced = False

    # ---------------------------------------------------------------- 主流程
    def push(self, block):
        """喂一块音频，返回这次切出来的句子列表（通常是空列表或一个元素）。"""
        a = _as_float_block(block)
        if a is None:
            return []

        cfg = self.config
        ms = a.size * 1000.0 / float(self.rate)

        self._level = rms(a)
        self._zcr = zero_crossing_rate(a)

        # 先按「上一块为止」的底噪算门限，再用它判断这一块。
        # 顺序不能反：如果先把自己的电平并进底噪窗口，
        # 一块响亮的语音会把门限抬到自己之上，然后它自己就被判成静音了。
        self._threshold = self._floor.threshold(self._scale)
        self._voiced = (self._level > self._threshold) and (self._zcr < cfg.zcr_max)

        # 底噪只在「不是人声」的时候更新。
        # 语音块如果也进窗口，连续说几句就会把底噪顶到语音电平，
        # 门限跟着一起涨，最后把正在说的话切断——句子越说越短。
        if not self._voiced:
            self._floor.update(self._level)

        out = []
        if self._voiced:
            if not self._active:
                # 起句：把预卷接上去，词首才不会被门限吃掉
                self._active = True
                self._buf = list(self._pre)
                self._pre = []
                self._pre_samples = 0
            self._silence_ms = 0.0
            self._buf.append(a)
            self._speech_ms += ms
            self._append_tail(a)
        else:
            if self._active:
                # 尾部静音也留着，识别器需要一点上下文才能确认说完
                if cfg.hang_blocks_keep:
                    self._buf.append(a)
                self._append_tail(a)
                self._silence_ms += ms
            else:
                self._append_pre(a)

        if self._should_cut():
            out.extend(self._cut())

        return out

    def flush(self):
        """会话结束时把还没切出去的音频吐出来。

        没有正在说的句子就返回空列表；正在说就当作一次强制切句。
        """
        if not self._active:
            self.reset()
            return []
        self._silence_ms = max(self._silence_ms, self.config.hang_ms)
        return self._cut()

    def partial(self):
        """流式临时结果：最近 stream_window_s 秒的音频，尾部裁到能量谷底。

        只用于「边说边出字」的显示，不用于最终结果。间隔不到
        stream_interval_s 就返回 None，让调用方跳过这次刷新。
        """
        cfg = self.config
        if not self._active:
            return None
        if self._speech_ms / 1000.0 < cfg.stream_min_s:
            return None
        now = self._clock()
        if now - self._last_partial < cfg.stream_interval_s:
            return None
        self._last_partial = now
        if not self._tail:
            return None
        seg = np.concatenate(self._tail) if len(self._tail) > 1 else self._tail[0]
        return trim_to_valley(seg, rate=self.rate, search_ms=cfg.valley_ms)

    # ---------------------------------------------------------------- 内部
    def _append_pre(self, a):
        """静音期滚动缓存，总量不超过 preroll_ms。"""
        limit = int(self.rate * self.config.preroll_ms / 1000.0)
        self._pre.append(a)
        self._pre_samples += a.size
        while self._pre and self._pre_samples > limit:
            dropped = self._pre.pop(0)
            self._pre_samples -= dropped.size
        # 保留窗口：整句最多 max_speech_ms，再长也只留后面一段
        hard = int(self.rate * (self.config.max_speech_ms + self.config.hang_ms) / 1000.0)
        if self._pre_samples > hard:
            self._pre = []
            self._pre_samples = 0

    def _append_tail(self, a):
        """流式窗口滚动缓存，总量不超过 stream_window_s。"""
        limit = int(self.rate * self.config.stream_window_s)
        self._tail.append(a)
        self._tail_samples += a.size
        while self._tail and self._tail_samples > limit:
            dropped = self._tail.pop(0)
            self._tail_samples -= dropped.size

    def _should_cut(self):
        if not self._active:
            return False
        cfg = self.config
        if self._silence_ms >= cfg.hang_ms:
            return True
        if self._speech_ms >= cfg.max_speech_ms:
            return True
        return False

    def _cut(self):
        cfg = self.config
        if not self._buf:
            self._active = False
            self._reset_sentence()
            return []

        seg = self._buf[0] if len(self._buf) == 1 else np.concatenate(self._buf)
        speech = self._speech_ms
        self._active = False
        self._reset_sentence()

        if seg.size == 0 or speech < cfg.min_speech_ms:
            return []

        seg = trim_tail_valley(seg, rate=self.rate, search_ms=cfg.valley_ms,
                               ratio=cfg.valley_ratio, pad_ms=cfg.valley_pad_ms,
                               min_ms=cfg.valley_min_ms)
        self._segments += 1
        return [seg]

    def _reset_sentence(self):
        self._buf = []
        self._tail = []
        self._tail_samples = 0
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._last_partial = -1e9

    def __repr__(self):
        return ("Segmenter(rate=%d, active=%s, speech=%.0fms, "
                "level=%.5f, thr=%.5f, segments=%d)"
                % (self.rate, self._active, self._speech_ms,
                   self._level, self._threshold, self._segments))
