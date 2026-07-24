# qtext vs black：case 级四象限分析（2026-07-16）

**问题**：ctrl 分支"保留图像+把问题换成通用提示 qtext"（What is the answer?）与默认的 black（黑图、保留问题）相比，训出的模型差在哪？

**对比对象**（同为 contrast-标准 × 默认6k数据 × Qwen3-VL-2B × step62，VLMEvalKit temp0/pp1.5/4096）：

| 模型 | BLINK | MMStar | MMBench | V* | MathVista | HR4K | HR8K | 7-bench | POPE | Hallu(3均) | ZoomBench |
|---|---|---|---|---|---|---|---|---|---|---|---|
| black（主线） | 58.71 | 62.53 | 77.06 | 76.96 | 67.90 | 77.62 | 73.75 | **70.65** | 89.13 | 54.40 | 43.67 |
| qtext | 55.13 | 60.40 | 74.48 | 72.77 | 64.90 | 75.00 | 73.25 | **67.99** | 88.93 | 51.70 | 43.20 |
| Δ(qtext−black) | −3.58 | −2.13 | −2.58 | −4.19 | −3.00 | −2.62 | −0.50 | **−2.66** | −0.20 | −2.70 | −0.47 |

（black 为 vllm_server 后端、qtext 为 offline 后端，聚合等价性已验证 ±1.5pp；逐题匹配率 ~85-90%，四象限计数含少量后端噪声，看净差方向即可。）

## 四象限（6 个 MCQ 基准，N=6356）

| 基准 | N | 双对 | 仅black对 | 仅qtext对 | 双错 | 净差 |
|---|---|---|---|---|---|---|
| BLINK | 1901 | 844 | 272 | 204 | 581 | **+68** |
| MMStar | 1500 | 777 | 161 | 129 | 433 | +32 |
| MMBench | 1164 | 808 | 89 | 59 | 208 | +30 |
| VStarBench | 191 | 130 | 17 | 9 | 35 | +8 |
| HRBench4K | 800 | 561 | 60 | 39 | 140 | +21 |
| HRBench8K | 800 | 532 | 58 | 54 | 156 | +4 |
| **合计** | 6356 | 3652 | **657 (10.3%)** | **494 (7.8%)** | 1553 | **+163** |

（MathVista 逐题需 fuzzy 匹配，近似串匹配下净差 +7 方向一致但计数不可靠，不计入合计。）

## 结论

1. **qtext 是全面小幅退化，不是某项能力的塌方**：没有任何基准 qtext 净赢；退化平摊在所有基准（HR8K 几乎持平 +4）。翻转是双向的（仅qtext对也有 7.8%），说明两个模型学到的东西大部分重叠、qtext 只是整体信号更弱。
2. **退化最集中的是"纯视觉对比/感知"类**：BLINK 的 Forensic_Detection（净+30/132）、Visual_Similarity（+19/135）两类合计贡献了 BLINK 净差的 72%；MMStar 的 instance reasoning / science&tech、HRBench 的 cross（跨区域对比）也偏高。这些恰是"不看图就完全没法答"的题。
3. **机制解释与 RA-VAD 设计自洽**：black 的 ctrl 分支彻底抹掉视觉内容，lp_hi−lp_ctrl 的对比信号 = "图像带来的全部增益"，权重集中在真正依赖视觉的 token 上；qtext 的 ctrl 保留了图像、只抹掉问题，对比信号变成"问题文本带来的增益"，视觉依赖 token 的对比度被稀释——所以受伤最重的正是纯视觉感知类题目。
4. **对论文的含义**：qtext 作为 ablation 支持"ctrl 分支必须移除视觉信息（而非文本信息）"的设计选择；black(内容抹除) > qtext(问题抹除)，7-bench −2.66pp、Hallu −2.70pp（ZoomBench 基本持平 −0.47，说明缩放类任务两者差异不大，退化集中在感知对比类）。

## 追加（2026-07-16 晚）：black+qtext 合体的潜力

**oracle union 上界**（任一模型对即算对，同 N 口径）：

| 基准 | black 单模 | 联合上界 | Δ |
|---|---|---|---|
| BLINK | 58.71 | 69.44 | +10.7 |
| MMStar | 62.53 | 71.13 | +8.6 |
| MMBench | 77.06 | 82.13 | +5.1 |
| VStarBench | 76.96 | 81.68 | +4.7 |
| HRBench4K | 77.62 | 82.50 | +4.9 |
| HRBench8K | 73.75 | 80.50 | +6.8 |
| **6-bench 平均** | **71.11** | **77.90** | **+6.8** |

Caveat：(a) 先知上界，真实集成（置信度路由/logit 平均）一般只能拿到 1/3~1/2，两模型投票破不了平局；
(b) 部分是噪声红利——两个不同 seed 的 black 做 union 预计也虚涨 ~4-5pp（可作对照实验，未实测）。

**建议的训练侧方案（待讨论）：双 ctrl 实验**——ctrl 分布取 black 与 qtext 两路的平均（或逐 token
对比权重取 max），让单模型同时吃"图像增益"与"问题增益"两种对比信号；90 步口径一次训练 ~2.5h 可验证。

数据来源：`VLMEvalKit/outputs_vllm_curated/contrast-standard-step62_server_.../normal_scoring/`（black）、`.../contrast_standard_qtext_step62_.../normal_scoring/`（qtext）、`logs/zoombench_qtext_step62_*.log`。
