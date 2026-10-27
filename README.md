# HAE-VAD Core

**Hearizon Audiology Engine — Voice Activity Detection**

自适应噪声底的流式语音活动检测与切句。Python 与 C++ 双实现，同一份输入逐位一致。

32 KB Python 包，或一个 17 KB 的 C++ 头文件。只依赖 numpy（C++ 版零依赖）。

---

## 为什么又写一个 VAD

因为现成的都不太好用在这个场景上：**实时字幕**。

实时字幕对切句的要求和语音识别、会议转录都不一样：

- 不能在句尾等太久。等 2 秒才出字，字幕就没意义了。
- 不能切在词中间。切掉一个 `/s/` 的开头，Spanish 就变成 fan。
- 门限不能写死。安静会议室里 0.008 是对的，开着空调的教室里 0.008 会把底噪当人声一直收。
- 不能只靠能量。风扇、空调、键盘的能量和轻声说话一样高，光看 RMS 分不开。

所以这里的做法是**门限跟着环境自动走**，而不是给你一个"灵敏度"滑块让你自己试。

## 四条和常见 VAD 不一样的地方

### 1. 门限从环境里估，不写死

维护一个 180 块的滑动窗口（30 ms 一块约 5.4 秒），取其中**第 20 分位**当底噪：

```
门限 = max(min_rms, 底噪 × floor_factor)
```

用低分位而不是均值，因为底噪的定义就是"一直存在的最低电平"——一声关门声不该把门限抬高五秒钟。

**关键细节：门限必须先算、后判断。** 顺序反了会出大问题：一块响亮的语音先被并进底噪窗口，门限瞬间抬到自己之上，然后它自己就被判成静音了。代码里这个顺序有注释，别改。

### 2. 过零率做第二道闸

白噪声、风噪、嘶声的过零率都在 0.45 以上，人声通常在 0.02–0.25。

```
判定为人声 = (RMS > 门限) and (过零率 < zcr_max)
```

`zcr_max` 默认 0.45 而不是 0.55：白噪声的过零率理论上就是 0.5，0.55 挡不住。代价是孤立的 `/s/` `/sh/` 可能被判成非人声，但它们很少单独成句，总有元音把整块的过零率拉下来。

### 3. 起句回补 300 ms

静音时也一直滚动缓存最近 300 ms，真正开口时把它接到句子前面。

这条不是可选优化，是修 bug 的必需品。`/s/` `/f/` 这类词首能量很低，会被门限切掉——**这就是 "Spanish" 被识别成 "fan" 的根因**。回补 300 ms 刚好够把词首救回来。

### 4. 尾音按能量谷底裁

切下来的片段尾部会带一段静音。对识别不是坏事（模型需要一点上下文才确认说完），但太长会拖慢。所以在尾部 320 ms 内找能量低于峰值 2% 的位置，从那里切开，再留 40 ms 余量防止切在字中间。

流式临时结果用另一套更轻的算法（`trim_to_valley`）：只按平均能量从尾部剥，不做逐窗搜索。

## 实测

60 秒「说 2 秒停 1 秒」的合成音频，16000 Hz，30 ms 一块：

| 预设 | 用时 | 实时倍数 | 切出 |
|---|---:|---:|---:|
| aggressive | 54.7 ms | 1096× | 20 句 |
| default | 54.8 ms | 1094× | 20 句 |
| far | 57.4 ms | 1044× | 20 句 |
| near | 58.0 ms | 1035× | 20 句 |
| noisy | 58.2 ms | 1031× | 20 句 |

切句延迟（语音结束到句子吐出，等于 `hang_ms` 加一块）：

| 预设 | 延迟 |
|---|---:|
| near | 690 ms |
| far | 870 ms |
| aggressive | 870 ms |

内存（单实例）：

| 项 | 占用 |
|---|---:|
| 底噪窗口（180 个 double） | 1.4 KB |
| 预录缓冲（300 ms） | 18.8 KB |
| Python 包合计 | 32 KB |
| C++ 头文件 | 16.7 KB |

复现：`python bench/bench.py --seconds 60`

## 快速开始

### Python

```bash
pip install numpy
```

把 `hae_vad/` 目录拷进项目，或者：

```bash
pip install -e .
```

实时麦克风（伪代码，录音部分用你自己的库）：

```python
import hae_vad

seg = hae_vad.Segmenter(rate=16000, config=hae_vad.FAR)

while True:
    chunk = mic.read(480)          # 30 ms
    for sentence in seg.push(chunk):
        pass_to_asr(sentence)      # 一句完整的音频

    partial = seg.partial()        # 临时结果；未到刷新间隔时是 None
    if partial is not None and partial.size:
        pass_to_asr(partial, final=False)
```

处理文件：

```python
import hae_vad

audio, rate = hae_vad.read_wav("meeting.wav")
seg = hae_vad.Segmenter(rate=rate, config=hae_vad.NEAR)

result = []
for i in range(0, audio.size, 480):
    result.extend(seg.push(audio[i:i + 480]))
result.extend(seg.flush())

for i, s in enumerate(result):
    hae_vad.write_wav(f"out/{i:03d}.wav", s, rate)
```

### 命令行

```bash
hae-vad meeting.wav                  # 切句并打印时间轴
hae-vad meeting.wav -p far           # 远场
hae-vad meeting.wav --export out/    # 每句导出成单独文件
hae-vad --print-presets              # 看有哪些预设
```

### C++

单头文件，C++17，只依赖标准库（`<vector>` `<deque>` `<algorithm>` `<cmath>`）。

```cpp
#include "hae_vad.h"

hae::vad::Segmenter seg(16000, hae::vad::presetFar());

// 每次回调拿到 30 ms 音频
std::vector<std::vector<float>> out = seg.push(buf, n);
for (auto& sentence : out) {
    feedAsr(sentence.data(), sentence.size());
}

auto partial = seg.partial();   // 临时结果
auto tail = seg.flush();        // 收尾
```

## 四种预设

| 预设 | 场景 | 关键差异 |
|---|---|---|
| `near` | 贴着电脑说话 | 默认参数，门限正常 |
| `far` | 四五米以上 | 门限 ×0.65，尾音等 900 ms，单句上限 15 秒 |
| `noisy` | 空调、风扇、多人 | 门限倍数 3.2，绝对下限 0.012，过零率放宽到 0.50 |
| `aggressive` | 极小声也要收 | 门限 ×0.45，绝对下限 0.005，代价是误触发变多 |

自定义：

```python
cfg = hae_vad.NEAR.with_(hang_ms=500.0, zcr_max=0.40)
seg = hae_vad.Segmenter(rate=16000, config=cfg)
```

## 参数

门限类：

| 字段 | 默认 | 作用 |
|---|---:|---|
| `min_rms` | 0.008 | 绝对下限，低于它一定是静音 |
| `zcr_max` | 0.45 | 过零率上限，挡噪声 |
| `floor_size` | 180 | 底噪窗口块数（30 ms 一块约 5.4 秒） |
| `floor_percentile` | 20.0 | 取第几分位当底噪 |
| `floor_factor` | 2.4 | 门限 = 底噪 × 倍数 |
| `far_field_scale` | 0.65 | 远场时门限再乘这个 |
| `force_loud_scale` | 0.45 | 强制模式门限再乘这个 |

节奏类：

| 字段 | 默认 | 作用 |
|---|---:|---|
| `block_ms` | 30.0 | 块长 |
| `hang_ms` | 700.0 | 静音超过这么久就切 |
| `min_speech_ms` | 400.0 | 语音短于这个丢弃（咳嗽、敲桌子） |
| `max_speech_ms` | 12000.0 | 语音超长强制切 |
| `preroll_ms` | 300.0 | 起句回补长度 |
| `hang_blocks_keep` | True | 尾部静音是否留在片段里 |

收尾裁切：

| 字段 | 默认 | 作用 |
|---|---:|---|
| `valley_ms` | 320.0 | 在尾部这么久内找谷底 |
| `valley_ratio` | 0.02 | 谷底能量低于峰值的比例 |
| `valley_pad_ms` | 40.0 | 谷底后留的余量 |
| `valley_min_ms` | 40.0 | 片段短于这么久不裁 |

流式：

| 字段 | 默认 | 作用 |
|---|---:|---|
| `stream_window_s` | 6.0 | 临时结果只看最近这么久 |
| `stream_interval_s` | 0.32 | 临时结果触发间隔 |
| `stream_min_s` | 0.30 | 语音短于这么久不出临时结果 |

## Python 和 C++ 逐位一致

`tests/run_parity.py` 会用同一段确定性音频分别跑两套实现，逐行比对：

```
$ python tests/run_parity.py
对拍通过：21 行逐行一致
```

比对的包括：资源生成（同一个 LCG 种子）、`rms` / `zero_crossing_rate` / `percentile`、四种预设下的完整切句结果（句子数 + 每句样本数）、`partial` 调用次数与末次长度、底噪估值、两种裁切函数。

需要装了 MSVC 的 Windows；没编译器时脚本会跳过而不是失败。

单测：

```bash
python -m unittest discover -s tests -v
# Ran 37 tests ... OK
```

## 和 HAE-NSG Core 的关系

两者同属 **Hearizon Audiology Engine**：

- **HAE-NSG Core**（[eyru/HAE-NSG-Core](https://github.com/eyru/HAE-NSG-Core)）：噪声抑制 + 增益，负责"听清楚"。
- **HAE-VAD Core**（本项目）：切句，负责"什么时候该转写"。

在 [Hearizon Sync](https://github.com/eyru/HearizonSync) 里是串联的：音频 → NSG（降噪、归一化）→ VAD（切句）→ 识别引擎。

## 限制

- **只支持 16 位 PCM wav**。要读 mp3/m4a 请先用 ffmpeg 转。
- **不检测音乐**。背景音乐的过零率可能落在人声范围内，会被当成说话。
- **不处理双讲**。两个方向同时说话时切句会来回跳。
- **过零率阈值对宽带人声偏保守**。嘶哑、气声重的声音可能被误判成非人声，需要调高 `zcr_max`。
- **C++ 版是 C++17，没有 C API 封装**。要从 C 调用需要自己加一层。

## 许可

MIT。见 [LICENSE](LICENSE)。

---

作者：何瑞程 · [Hearizon Sync](https://github.com/eyru/HearizonSync) 的一部分
