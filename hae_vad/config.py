"""切句参数。

所有能影响切句行为的数字都集中在这里，Segmenter 本身不存魔法数。
改参数请构造新的 VadConfig，不要改模块级常量。
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class VadConfig:
    """一组完整的切句参数。

    字段分三类：门限（怎么判断「有声音」）、节奏（什么时候切句）、
    收尾（切下来的片段怎么裁）。
    """

    # ---- 门限 ----
    min_rms: float = 0.008
    """绝对下限。RMS 低于它的块一定算静音，不管环境多安静。"""

    zcr_max: float = 0.45
    """过零率上限。嘶声、风噪、白噪声的过零率接近 0.5，人声通常在 0.02-0.25，
    用它在能量门限之外再挡一道。

    取 0.45 而不是 0.55：白噪声的过零率理论上就是 0.5，0.55 挡不住它。
    代价是孤立的高频摩擦音（/s/ /sh/）可能被判成非人声，但它们很少单独成句，
    总有元音把整块的过零率拉下来。
    """

    floor_size: int = 180
    """噪声底滑动窗口的样本数。30 ms 一块时，180 块约 5.4 秒。"""

    floor_percentile: float = 20.0
    """取窗口里的第几分位当底噪。用低分位而不是均值，避免被偶发响声带偏。"""

    floor_factor: float = 2.4
    """动态门限 = max(min_rms, 底噪 × 这个倍数)。环境越吵门限越高。"""

    far_field_scale: float = 0.65
    """远场时把门限再乘这个系数。人声到麦克风已经很弱，门限必须放低。"""

    force_loud_scale: float = 0.45
    """强制放大时同理，比远场更激进。"""

    # ---- 节奏 ----
    block_ms: float = 30.0
    """每次喂进来的音频块长度。库按这个粒度累计时长。"""

    hang_ms: float = 700.0
    """尾部静音超过这么久就认为一句话说完了。"""

    min_speech_ms: float = 400.0
    """累计语音短于这么久不认为是句子，丢掉（咳嗽、敲桌子）。"""

    max_speech_ms: float = 12000.0
    """累计语音超过这么久强制切断，防止有人一直说不停。"""

    preroll_ms: float = 300.0
    """起句回补长度。静音时也滚动缓存这么长，等真正开口时接到前面。

    这一条是必需的：/s/ /f/ 这类词首能量很低，会被门限切掉，
    表现为「Spanish」被识别成「fan」。回补 300 ms 就够把词首救回来。
    """

    hang_blocks_keep: bool = True
    """切句时是否把尾部静音一起留在片段里。

    保留一点尾部静音对识别更友好（模型需要一点上下文才能确认说完），
    多余的静音由 trim_tail_valley 再裁掉。
    """

    # ---- 收尾裁切 ----
    valley_ms: float = 320.0
    """在片段尾部这么长的范围内找能量谷底。"""

    valley_ratio: float = 0.02
    """谷底判据：窗口能量低于峰值的这个比例。"""

    valley_pad_ms: float = 40.0
    """找到谷底后再往后留一点余量，防止切在字中间。"""

    valley_min_ms: float = 40.0
    """片段短于这么久就不裁，免得把整句裁没。"""

    # ---- 流式 ----
    stream_window_s: float = 6.0
    """流式临时结果每次只取最近这么久。"""

    stream_interval_s: float = 0.32
    """流式临时结果的触发间隔。"""

    stream_min_s: float = 0.30
    """累计语音短于这么久不出临时结果。"""

    def with_(self, **kwargs):
        """返回改了若干字段的新配置。frozen dataclass 不能直接改，用这个。"""
        unknown = set(kwargs) - set(self.__dataclass_fields__)
        if unknown:
            raise TypeError("未知的配置字段：%s" % ", ".join(sorted(unknown)))
        return replace(self, **kwargs)


#: 默认配置，适合笔记本内置麦克风、安静房间。
DEFAULT = VadConfig()

#: 近讲：贴着电脑说话，门限可以正常。
NEAR = VadConfig()

#: 远场：四五米以上。门限放宽，尾音留长一点，避免把慢速说话的停顿切断。
FAR = VadConfig(far_field_scale=0.65, hang_ms=900.0, max_speech_ms=15000.0)

#: 嘈杂环境：空调、风扇、多人。门限抬高，过零率收紧。
NOISY = VadConfig(floor_factor=3.2, min_rms=0.012, zcr_max=0.50)

#: 强制模式：极小声也要收进来，代价是误触发变多。
AGGRESSIVE = VadConfig(far_field_scale=0.45, min_rms=0.005,
                       hang_ms=900.0, max_speech_ms=15000.0)

_PRESETS = {
    "near": NEAR,
    "far": FAR,
    "noisy": NOISY,
    "aggressive": AGGRESSIVE,
    "default": DEFAULT,
}


def get(name):
    """按名字取预设。名字不认识就抛 ValueError，不静默回退。"""
    try:
        return _PRESETS[str(name).strip().lower()]
    except KeyError:
        raise ValueError(
            "未知的 preset：%r，可选 %s" % (name, ", ".join(sorted(_PRESETS)))
        ) from None


def names():
    """返回所有预设名（排序后）。"""
    return sorted(_PRESETS)
