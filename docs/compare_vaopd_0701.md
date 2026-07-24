# OPSD / VA-OPD Diagnostics + Reproduction: Plan & Results

最后更新:2026-07-07。每个 Phase 下分"计划"和"结果"两块;没跑的标记 ⏳ 待执行,跑完就把结果贴进对应位置,保持这一份文档是计划和结果的唯一入口。

## 评测口径锁定说明(2026-07-07,重要,影响下面所有 ZoomBench 数字的解读)

**ZoomBench 正式结果统一锁定为 Phase 4 双 bug 修复后的 native eval pipeline("v2-fixed")**,不再用此前的"官方 pipeline"(`run_official_eval.sh`,thinking + 未修复 judge)作为主口径。原因:官方 pipeline 虽然数字看起来更稳定,但 Phase 4 排查发现 native pipeline 存在两个可定位、可修复的 judge bug(MCQ 快速通道漏掉 zoombench、选项提取抓错方向),这两个 bug 修复后 native pipeline 才是当前更可信的判分方式;而"官方 pipeline"本身没有被同等严格地审计过,不能默认它没有类似问题。

**这个决定的直接后果**:此前所有标注"官方 pipeline"的 ZoomBench 数字(noimg=37.16、visionopd=40.00、black=36.33、degrade=36.80、qvis=36.69、2B base=37.40、4B base=41.07)**降级为历史参照,不再是正式表格口径**。当晚又发现并修复了第三个 judge bug(数字题长 CoT 提取失败,见 Phase 4 Bug 3),口径已从 v2-fixed 升级为 **v3-fixed**。目前已有 v3-fixed 数字的 checkpoint:2B/4B base、noimg-2B、noimg-4B、VisionOPD-2B(full)、VisionOPD-4B、noimg-virl39k-2B(step1197)、noimg-vision-sr1-2B(step1480)(见 Phase 3.1 / Phase 4)。**black/degrade/qvis 三个变体目前还没有 v3-fixed 的 ZoomBench 数字,是当前最优先的待办**,在补上之前,主表 ZoomBench 列这几格保持"待重判",8-benchmark composite 表的相关数字也是暂时性的、需要重算。

## 进度总览

| Phase | 内容 | 状态 |
|---|---|---|
| -0.5 | 四种 control 信号能否互补融合 | ✅ 已出结果(2026-07-01) |
| -0.3 | qvis 修正版(mask)验证性实验 | ✅ 已出结果(2026-07-02) |
| 0 | 负 token 事后分类(A)+ answer-leakage 量级检查(B) | ✅ 已出结果(2026-07-01) |
| 0.8a | noimg eval-level 随机性检查(同 checkpoint,换解码参数) | ✅ 已出结果(2026-07-02) |
| 0.8b | noimg training-level 种子稳健性检查(换训练种子重训) | ✅ 已出结果(2026-07-06):composite 稳健(-0.38),但单项噪声不小(最高 -2.26pp) |
| 1 | VA-OPD 加权/散度机制 Stage 1 训练 | ✅ 已出结果(2026-07-05):5 组全部完成,7-benchmark+ZoomBench(官方 pipeline,待用 v2-fixed 重判)评测全部完成 |
| 1.5 | Stage 1 结果验证性检查 | ✅ 已出结果(2026-07-05) |
| 2 | 训练机制探索:三组件贡献分解(纯EMA/shuffle/完整)+ GRPO compute-matched + GRPO+RA-VAD 结合 | 🔄 进行中,见下 |
| 3 | ViRL39K / Vision-SR1 外部数据集迁移(同 noimg schema,换训练数据) | ✅ 已出结果(2026-07-07),ZoomBench 已用 v3-fixed pipeline 重判(virl39k 38.93%,sr1 39.64%,和旧数几乎无变化) |
| 4 | native eval judge 三 bug 排查与修复(现为 ZoomBench 正式口径) | ✅ 已出结果(2026-07-07) |

---

## Vision-OPD 2B 已完成评测汇总

本节合并 `docs/vision_opd_eval_record_20260701.md` 里已经跑完的 Vision-OPD 2B 系列结果,作为主参照表。除特别说明外,VLMEvalKit 7-benchmark 结果使用 `temperature=0.0`, `max_new_tokens=4096`, judge 为 `gpt-5.4-mini-2026-03-17`;ZoomBench 见上面的口径锁定说明,统一用 v2-fixed native pipeline。

### 参与比较的 checkpoint

- `Vision-OPD-baseline-Qwen3-VL-2B-Instruct/global_step_65`
- `Vision-OPD-noimg-Qwen3-VL-2B-Instruct/global_step_62`
- `Vision-OPD-black-Qwen3-VL-2B-Instruct/global_step_65`
- `Vision-OPD-degrade-Qwen3-VL-2B-Instruct/global_step_62`
- `Vision-OPD-qvis-Qwen3-VL-2B-Instruct/global_step_62`
- `Vision-OPD-visionopd-Qwen3-VL-2B-Instruct/global_step_65`

### Consolidated benchmark scores

`visionopd` 行来自 strict/rejudge 路径;其余行来自 `normal_scoring`。

| model | BLINK | MMStar | MMBench_DEV_EN | VStarBench | MathVista_MINI | HRBench4K | HRBench8K | ZoomBench(v3-fixed native) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 55.44 | 58.87 | 69.24 | 72.25 | 61.80 | 73.25 | 69.63 | — |
| noimg | 55.50 | 61.20 | 76.55 | 73.82 | 63.20 | 75.75 | 71.38 | **36.45** |
| black | 54.71 | 60.33 | 76.46 | 73.82 | 63.60 | 75.25 | 70.63 | **36.33** |
| degrade | 54.45 | 61.87 | 75.09 | 74.35 | 64.90 | 75.13 | 71.75 | **36.33** |
| qvis | 53.34 | 61.20 | 75.43 | 71.20 | 62.70 | 75.25 | 71.25 | **38.11** |

**✅ v3-fixed 重判已全部完成(2026-07-08,零训练成本,只重跑判分不重推理)**。至此本表 ZoomBench 列 6 个变体(noimg/black/degrade/qvis/visionopd/两个base)全部是统一 v3-fixed 口径,可以直接横向比较,不再有"待重判"占位。观察:qvis(38.11)是四个 RA-VAD 变体里 ZoomBench 最高的,黑箱/降质两个变体(black/degrade)完全打平在 36.33,noimg(36.45)略高于两者、明显低于 qvis。四个变体全部低于 2B base(42.49),幅度在 4.4-6.2pp 之间,和 noimg/visionopd 已经确认的"相对 base 退步"模式一致——RA-VAD 系列变体在 ZoomBench 上普遍不如 base,程度相近,不是 noimg 特别差。
| visionopd(full) | 47.55 | 58.60 | 75.52 | 75.39 | 61.10 | 74.88 | 70.38 | **37.51** |

参照(v3-fixed native,845/845):Qwen3-VL-2B base = 42.49%,Qwen3-VL-4B base = 44.14%。

> **口径版本说明(2026-07-07 晚间更新)**:上面 v2-fixed 之后又发现并修复了**第三个 judge bug**(见 Phase 4 Bug 3:开放式数字题的长 CoT 提取失败,系统性漏判"话多"的模型),下表已改为最新的 **v3-fixed** 数字。v2→v3 变化幅度不小(noimg 31.72→36.45,visionopd(full) 35.50→37.51),说明 v2 时点的叙事已经过时,以本节为准。

**关键观察(锁定 v3-fixed pipeline 后)**:noimg(36.45)和 visionopd(full)(37.51)都**仍然低于** 2B base(42.49)——noimg 掉 6.04 分,visionopd(full) 掉 4.98 分,**两种方法在 ZoomBench 上相对 base 都是退步**,但幅度比 v2 时点(-11.12/-7.34)小了近一半,相当一部分之前的"退步"是 Bug 3(长 CoT 数字题误判)造成的假象。**相对排序不变**:visionopd(full) 依然比 noimg 更接近 base(掉得更少,1.06pp 的差距),但两者已经非常接近,"crop-based 方法在 search 类任务上有相对优势"这个结论现在证据强度比 v2 时点弱一些,不宜过度强调优势幅度。

**这对叙事的影响**:样本级复核(见 Phase 4 "noimg 为什么比 base 差" 分析)显示,noimg 真实的退步集中在**需要高精度读数的任务**(时钟指针角度、精确计数)而非全面感知能力下降——这比"灾难性遗忘/窄域过拟合"的笼统解释更精确、更可证伪,建议论文里采用样本级证据而不是笼统的 composite 分数下降来解释这个现象。

**为什么这仍然不代表方法缺陷**:HRBench(同样高分辨率场景,但不需要主动裁剪)noimg 依然明显优于 base 和 visionopd,说明"静态高分辨率感知"和"主动视觉搜索/精细读数"是两种不同能力,noimg 提升的是前者,在后者上有可解释、局部化的退步(集中在钟表/计数类),不是随机噪声或方法全面失效。

**8-benchmark composite 表(下方)目前仍基于旧的官方 pipeline ZoomBench 数字,是暂时性的,black/degrade/qvis 补上 v3-fixed 数字后需要重新计算。**

### noimg vs VisionOPD:8-benchmark 综合平均分(2026-07-05,⚠️ 基于旧官方 pipeline ZoomBench 口径,待重算)

| model | 8-benchmark 平均(含官方 pipeline ZoomBench,待重算) |
|---|---:|
| **noimg** | **64.32** |
| degrade | 64.29 |
| black | 63.89 |
| qvis | 63.38 |
| visionopd | 62.93 |

**论证策略(不受 pipeline 切换影响,仍然成立)**:不需要证明"`noimg` 在 ZoomBench 上也更好"——VisionOPD 靠 bbox-crop teacher 对 search/zoom 类任务做了针对性优化,代价是别处掉分。`noimg` 不需要任何 benchmark-specific 信号,在 7 个非 search 的感知 benchmark 上全面或大部分领先。**但上面这张 composite 表的具体数字,在 black/degrade/qvis/visionopd 补上 v2-fixed ZoomBench 之前不能直接引用,只能引用 7-benchmark(不含 ZoomBench)的平均分作为当前唯一站得住的 composite 指标。**

### noimg vs 纯 GRPO baseline(2026-07-06)

额外跑了一个 vanilla GRPO baseline(`grpo_vanilla_qwen3vl2b_3ep`,2B,3 epoch / 585 步,无 RA-VAD/OPSD 机制,纯 GRPO reward 训练),VLMEvalKit 7-benchmark 结果:

| Benchmark | noimg(62步/1epoch) | grpo_vanilla_3ep(585步) |
|---|---:|---:|
| BLINK | 55.50 | 53.76 |
| MMStar | 61.20 | 62.73 |
| MMBench_DEV_EN | 76.55 | 79.38 |
| VStarBench | 73.82 | 80.10 |
| MathVista_MINI | 63.20 | 68.00 |
| HRBench4K | 75.75 | 74.38 |
| HRBench8K | 71.38 | 73.00 |
| **7-benchmark 平均(不含 ZoomBench)** | 68.20 | **70.19** |

⚠️ **不是公平对比,不能直接下结论**:GRPO baseline 训练了 585 步/3 epoch,用的数据是 `data/train.parquet`;`noimg` 只训练了 62 步/1 epoch,用的是 `data/train_answer.parquet`——训练量差了近 9.4 倍,训练数据也不同。见下方 **Phase 2** 里的 compute-matched 对比,这是当前最紧急的训练类待办事项。

### 直接观察

- `noimg` 是当前 2B RA-VAD 变体里 7-benchmark(不含 ZoomBench)最稳的主 control。
- `degrade` 在 MathVista_MINI、VStarBench、HRBench8K 上最好,但更像细粒度 visual acuity signal,不能直接替代 noimg。
- `qvis` 更适合作为 task-conditioning/noise-filter 诊断,不适合作为主 visual-dependence control。
- `visionopd` 在 VStarBench 最好,ZoomBench(v2-fixed)相对 base 掉得比 noimg 少,但 BLINK/MMStar/MathVista 明显更弱,不是当前 Stage 1 的最佳统一基线。

---

## Phase -0.5: 四种 control 信号能否互补融合 ✅

### 背景判断

四种 control 测的不是同一个构念的四份带噪拷贝,而是不同的轴:

- **noimg / black**:测"这个 token 有没有依赖视觉内容"(existence)
- **degrade**:测"已确定依赖视觉的前提下,细节够不够"(sufficiency,对应 vlm_plan_v1.md 里 P0-A 的 acuity 轴)
- **qvis**:测"这个 token 对当前任务/prompt 的依赖程度",但没把任务依赖和视觉依赖分开,混了格式/选项类噪声

### 结果(2026-07-01)

固定同一批 `Vision-OPD-qvis-Qwen3-VL-2B-Instruct` rollout response,离线重算四种 control 的 `ra_raw`/`ra_weight`。

**Jaccard / correlation 结果**:

| Pair | Jaccard | raw corr | weight corr | 解释 |
|---|---:|---:|---:|---|
| noimg-black | 0.633 | 0.721 | 0.685 | 最接近,都主要测视觉内容存在性 |
| noimg-degrade | 0.468 | 0.218 | 0.323 | 重叠明显更低,degrade 提供细节/acuity 轴的补充 |
| qvis-noimg | 0.488 | 0.100 | 0.148 | qvis 和视觉依赖信号弱相关 |
| qvis-black | 0.571 | 0.056 | 0.112 | qvis raw gap 和 black 基本不相关 |

**融合假设统计**:Core(noimg∩black)= 47.1% 全部 token,覆盖 noimg/black positive 各 78.3%;Detail-augment(Core∪degrade)= 73.1%;qvis positive \ noimg positive = 41.1%。

**当前决策**:不建议 naive fusion;若加对照组优先试 Core;若验证互补收益再试 Core∪normalized(degrade)(必须归一化,不能直接 union——见 Phase 1.5 检查项 1);qvis 保留为诊断/过滤器。

### Spatial-scale 10% degrade 诊断(2026-07-02)

10% spatial-scale 明显增强了 degrade 信号(positive mass/token 约为原 degrade 2.2x,mean `ra_raw` 约 3.2x),说明原 degrade 确实偏 mild。但仍远弱于 noimg(约 25%/17%),不是替代品。建议放进 Detail-augment 路线,不替代 Core。

### Same-Response Paired Control Comparison

固定 qvis rollout 的 response token 做四组精细对比,补充 Argmax by token(qvis 49.3% raw / 45.9% weight 最强,degrade 归一化后从 8.3% 升到 19.2%,noimg 从 32.8% 降到 21.6%)和 Sign agreement(noimg-black 75.7% 同号最高,qvis-noimg 只有 52.4%)。

**结论**:四种 control 不能互换。noimg 仍是干净视觉 token 选择的最佳主 control。

---

## Phase -0.3: qvis 修正版验证性实验 ✅

### 背景

qvis 和 noimg 的 sign agreement 只有 52.4%,问题出在 control 换了整个任务,引入了任务格式差异。

### 结果(2026-07-02)

`qvis_mask`(保留任务格式,只 mask 问题内容词)把 positive_ratio 从 68.5% 降到 53.5%,但和 noimg 的 same-sign 只有 55.2%,`ra_raw` corr 从 0.109 降到 0.036,没有变得更可靠。`I+Q_vs_I`(完全去掉问题)同样没有改善。

**决策**:`qvis_mask`/`I+Q_vs_I` 均不加入训练矩阵;qvis 相关方向仅作诊断/噪声过滤用。

---

## Phase 0: 两个零成本诊断实验 ✅

### 结果(2026-07-01)

固定 121 条匹配 response,五个 mode(qvis/noimg/black/degrade/answer_hint)全量统计。

**诊断 A**:负 token 占比对 correctness 的预测力不强(qvis -0.076、noimg +0.017、black -0.277、degrade -0.217)。

**诊断 B**:answer-hint 泄露的 gap 规模(abs_mass/token 0.1250)接近 degrade(0.1089),远低于 qvis/noimg/black,支持"RA-VAD gap 不是在拟合泄露"。

**决策**:暂不加 signed/negative-token 变体;这条结论目前只在 noimg 语境下站得住,若扩展到 black/degrade 需重新验证。

---

## Phase 0.8a: noimg eval-level 随机性检查 ✅

用同一 checkpoint(`global_step_62`)换解码参数独立重跑 MMStar/VStarBench,排除评测阶段的运气。

| setting | MMStar | VStarBench |
|---|---:|---:|
| 原记录 noimg | 61.20 | 73.82 |
| rerun,temp=0.0 | 61.40 | 73.82 |
| rerun,temp=0.7 | 61.60 | 72.25 |

**重要方法论说明**:VLMEvalKit 的 `seed` 没有传到 vLLM engine(恒为 `seed=0`),这次严格说是独立重跑+temperature 消融,不是 seed 控制实验。

**结论**:temp=0 下分数可复现,说明**这个 checkpoint 的分数不是评测运气**。temp=0.7 下 VStarBench 回落到 baseline 水平,正式表格必须锁定 temperature=0.0。**这一步不能证明"noimg 训练配置本身稳健"**——全程没有重新训练,这个问题留给 Phase 0.8b。

**补充:5-benchmark 独立重跑(同一 `global_step_62` checkpoint,temp=0.0,2026-07-02)**

不只 MMStar/VStarBench,把同一 checkpoint 能确认的另外几个 benchmark 也独立重跑了一遍(BLINK/HRBench4K/HRBench8K,经 config 里的 `model_path` 核实确实指向原始 noimg checkpoint):

| Benchmark | 原记录 noimg | 独立重跑(同一 checkpoint) | 差值 |
|---|---:|---:|---:|
| BLINK | 55.50 | 55.39 | -0.11 |
| MMStar | 61.20 | 61.40 | +0.20 |
| VStarBench | 73.82 | 73.82 | 0.00 |
| HRBench4K | 75.75 | 75.88 | +0.13 |
| HRBench8K | 71.38 | 71.13 | -0.25 |

全部差值都在 ±0.3 以内,坐实"temp=0 下 noimg 的分数是可复现的、不是评测运气"。**结论边界不变**:仍然是同一个 checkpoint 的独立评测重跑,不是重新训练,不能替代 Phase 0.8b 的训练种子稳健性检查。

> **勘误(2026-07-06)**:此前这里的表格误把 MMBench_DEV_EN/MathVista_MINI/ZoomBench 三项数字也标成了"同一 checkpoint 独立重跑",经核对 VLMEvalKit 输出目录里 `configs/*.json` 的 `model_path` 字段,这三项实际来自 Phase 0.8b **重新训练**的 `Vision-OPD-noimg-seed123-Qwen3-VL-2B-Instruct` checkpoint,已移到下面 Phase 0.8b 表格里,不再算在本节内。

---

## Phase 0.8b: noimg training-level 种子稳健性检查 ✅

"noimg 是四组里综合最好"(以及 Phase -0.5 的 Core-vs-noimg 决策)目前都建立在 noimg/black 各自只跑过一次训练这个前提上。用完全相同配置、只换训练 random seed,完整重训一次 noimg(`Vision-OPD-noimg-seed123-Qwen3-VL-2B-Instruct`,global_step_62,训练耗时约 10 小时,Jul 2 完成),8-benchmark 评测对比新旧 seed 的差值。

**结果(2026-07-06,VLMEvalKit 7 项 + ZoomBench 官方 pipeline【此列为旧口径,待用 v2-fixed 重判】,全部 temp=0.0)**:

| Benchmark | noimg(原始训练,seed 默认) | noimg-seed123(独立重训) | 差值 |
|---|---:|---:|---:|
| BLINK | 55.50 | 53.24 | -2.26 |
| MMStar | 61.20 | 61.93 | +0.73 |
| MMBench_DEV_EN | 76.55 | 76.37 | -0.18 |
| VStarBench | 73.82 | 74.35 | +0.53 |
| MathVista_MINI | 63.20 | 64.10 | +0.90 |
| HRBench4K | 75.75 | 74.50 | -1.25 |
| HRBench8K | 71.38 | 70.00 | -1.38 |
| ZoomBench(官方 pipeline,旧口径) | 37.16 | 37.04 | -0.12 |
| **8-benchmark 平均(旧口径)** | **64.32** | **63.94** | **-0.38** |

**结论**:7-benchmark(不含 ZoomBench)层面 noimg 配置对训练种子是稳健的。**单项 benchmark 层面的种子噪声不小**——BLINK 摆动 -2.26pp,HRBench8K -1.38pp,HRBench4K -1.25pp,量级明显大于 Phase 0.8a 纯评测重跑的噪声(±0.3pp 以内),说明这部分差异确实来自训练随机性,不是评测阶段的运气。

**这份噪声本底的用途**:上面 Phase 1 的"结果的噪声本底重新校验"一节,直接用这张表的 `|差值|` 逐 benchmark 校验了 5 组 Stage 1 结果(7-benchmark 部分),得出了比 composite 层面更精确的判断(见 Phase 1 结果段落)。ZoomBench 列因为 pipeline 口径已变,不再用于噪声校验,若需要,应等 `noimg-seed123` 补上 v2-fixed ZoomBench 数字后重新计算。

**预计工时**:训练量级和现有单个 noimg run 相当(已完成)。

---

## Phase 1(已收紧范围): 只换 VA-OPD 的加权机制,不换 teacher、不新增 control 类型 ✅

**范围边界**:teacher 用现有 EMA self-teacher;control 用现有四选一;只替换 `ra_vad.py` 里 `ra_pos_t` 到 `w_t` 的下游构造方式。

### 实验矩阵(共 5 组,均在 noimg 上验证)

| 变体 | control | weighting_mode | rollout_reweight | 散度 |
|---|---|---|---|---|
| 现状对照(共享基准) | noimg | continuous | False | forward KL |
| VA-OPD token 机制 | noimg | vaopd_grouped | False | forward KL |
| VA-OPD 全机制 | noimg | vaopd_grouped | True | forward KL |
| 散度对照:reverse | noimg | continuous | False | reverse KL |
| 散度对照:JSD | noimg | continuous | False | JSD α=0.5 |

### 结果(2026-07-05):5 组全部完成 + ZoomBench(官方 pipeline,旧口径,待 v2-fixed 重判)

| Benchmark | noimg(基准) | vaopd-grouped-forward | vaopd-grouped-rollout-forward | continuous-reverse | continuous-jsd |
|---|---:|---:|---:|---:|---:|
| BLINK | 55.50 | 55.08 | 52.55 | 55.18 | 53.87 |
| MMStar | 61.20 | 60.60 | 59.80 | 60.80 | 60.07 |
| MMBench_DEV_EN | 76.55 | 76.37 | 75.26 | 76.89 | 75.43 |
| VStarBench | 73.82 | 73.30 | 72.25 | 73.30 | 74.87 |
| MathVista_MINI | 63.20 | 61.60 | 62.30 | 62.70 | 63.10 |
| HRBench4K | 75.75 | 75.25 | 73.13 | 74.88 | 74.63 |
| HRBench8K | 71.38 | 70.63 | 67.75 | 71.50 | 70.38 |
| ZoomBench(官方,旧口径) | 37.16 | 37.16 | 37.16 | 37.04 | 36.33 |
| **8-benchmark 平均(旧口径)** | **64.32** | 63.75 | 62.53 | 64.04 | 63.59 |

**初步结论(7-benchmark 层面,不受 pipeline 切换影响)**:硬分组 token 加权(`vaopd_grouped`)相对现状连续加权是轻微退步,加上 rollout reweight 后退步更明显;散度轴上 reverse KL 基本持平,JSD 轻微退步但 VStarBench 有独特优势。**权重机制维持现状连续加权、散度可选 reverse KL** 是目前综合最稳的组合,4 个新变体没有一个在 7-benchmark 平均分上超过 noimg 基准。ZoomBench 列和 composite 平均待用 v2-fixed pipeline 重判后更新。

### 结果的噪声本底重新校验(结合 Phase 0.8b,2026-07-06,基于 7-benchmark,不含 ZoomBench)

Phase 0.8b 测出了 `noimg` 换训练 seed 后的逐 benchmark 差值,可以直接当噪声本底,重新逐项校验上表每个变体相对 noimg 基准的差值是不是真实效应,而不是只看 composite 层面:

| 变体 | 超出噪声本底的 benchmark 数(7项) | 具体判断 |
|---|---:|---|
| vaopd-grouped-forward | 1/7(仅 MathVista,-1.60 vs 噪声 0.90) | 之前"普遍低 0.3-1.9 点"的表述不成立,6/7 项在噪声范围内,唯一疑似真实效应是 MathVista 退步 |
| vaopd-grouped-rollout-forward | 6/7 | 全面、真实的退步,不只是"普遍较弱"的定性判断,是有统计意义支撑的结论,这条最稳固 |
| continuous-reverse | 1/7(MMBench +0.34,略超其极窄噪声本底 0.18,绝对幅度很小) | 7 项里几乎全部在噪声范围内,和 noimg 基准统计上无法区分,不是"可能是噪声"而是"没有证据显示真实差异" |
| continuous-jsd | 3/7(MMStar/MMBench 真实退步,VStarBench +1.05 真实提升) | 有取舍的真实 trade-off,VStarBench 优势是真实的不是噪声 |

**方法论说明**:噪声本底来自 `noimg` 自己换 seed 的方差,拿去校验其他四个变体(不只换了 seed,还换了 loss 机制)的差值,隐含假设是不同 loss 配置的训练稳定性/方差量级相近——这是合理的一阶近似,不是严格验证过的。如果要做到完全严谨,理想情况下应给 `vaopd-grouped-rollout-forward`(效应最大)也补一个 seed 确认,但它的效应量是噪声本底的 2-4 倍,确认的优先级明显低于 Phase 2b 的 GRPO 对比。

**更新后的结论**:权重机制维持现状连续加权(硬分组本身没有站得住的负面证据,但加了 rollout reweight 后的全机制版本是明确、真实的退步,不建议采用);散度选择上,reverse KL 和现状基本无法区分,可以作为等价备选;JSD 不建议作为默认选择,净效应不划算。

---

## Phase 1.5: Stage 1 结果验证性检查 ✅

**检查项 1(Detail-augment 选择性稀释)**:不适用,当前 Stage 1 全部固定 noimg,没有训练 Core∪degrade 组合,留给 Stage 2。

**检查项 2(归一化放大 artifact 检查,2026-07-05)**:degrade 高权重 token(≥p75/p90/p95)与 noimg/black 判负的重叠率(23-29%)都明显低于全体 token 基线(43.7%/47.0%),flip-in token(归一化前非最强、归一化后变最强)同样低于基线,genuine token(归一化前后都最强)重叠率反而更高(81.5%/77.6%),符合"degrade 测细粒度细节"的定性。**结论**:归一化放大是真实的,但没有证据表明放大集中在噪声 token 上;Phase 1 的 5 组结果(全部基于 noimg,不涉及 degrade)不受影响,可直接使用。

---

## Phase 2: 训练机制探索——三组件贡献分解 / GRPO compute-matched / GRPO+RA-VAD 结合 🔄

### 背景

当前 RA-VAD 机制的一个关键特征:**没有外部信息注入**——没有比 student 更强的 teacher(EMA self-teacher)、没有额外输入(high-info 条件下 teacher/student 看的是同一张原图同一个问题)、没有 ground-truth 正确性标签。唯一的信号来源是"拿掉图像后,模型自己对同一个 token 的置信度会不会掉"。这和 VA-OPD(外部更大 teacher)、VOPD(teacher 看 bbox crop)、GRPO(靠外部 verifiable reward)在机制类别上都不是一类东西,更接近 self-supervised 里"用模型自身对扰动的敏感度构造训练信号"这一类方法。这解释了为什么 RA-VAD 的提升幅度天然有限,也是下面待办实验的共同出发点。

### Phase 2-核心:三组件贡献分解(与 2b 并列为成稿前必须完成的核心消融)

**动机**:审稿人必问"到底是 noimg 的视觉相关信号起作用,还是 EMA self-teacher 起作用"。拆成三个正交变量:

1. **EMA self-teacher 本身**:哪怕不做 token 加权,单纯用 EMA 做自蒸馏可能就有正则化收益。
2. **加权动作本身**:任何非均匀加权都可能有 regularize 效果,不管权重是不是视觉相关。
3. **权重具体来自"视觉相关性"这个信号**。

当前 claim 隐含"是 3 起作用",需要三组对照拼成贡献分解表:

| 变体 | EMA teacher | token 加权 | 权重来自视觉相关性 | 隔离的问题 | 状态 |
|---|:-:|:-:|:-:|---|---|
| base model | ✗ | ✗ | ✗ | 起点 | ✅ 已有(7-bench 平均 65.78) |
| **对照组 X:纯 EMA(均匀权重)** | ✓ | ✗(w_t=1) | — | EMA 自蒸馏单独值多少 | ✅ 训练+评测已完成(2026-07-08,`Vision-OPD-noimg-uniform-X-Qwen3-VL-2B-Instruct/global_step_62`)。**7-bench 平均 67.79** |
| **shuffle(原 Phase 2a)** | ✓ | ✓ | ✗(打乱) | 加权动作 vs 视觉信号 | 代码已实现(2026-07-07):`ra_weighting_mode=shuffled_control`,逐行随机置换真实 `ra_weight` 的 token↔权重对应,保持同 row 权重分布不变,独立 `ra_shuffle_seed`;单元测试已验证。✅ 训练已完成(2026-07-08,`Vision-OPD-noimg-shuffled-control-Qwen3-VL-2B-Instruct/global_step_62`,已 merge)。🔄 VLMEvalKit 7-benchmark 评测跑中(GPU1,推理阶段),结果待补 |
| **noimg(完整方法)** | ✓ | ✓ | ✓ | 完整 | ✅ 已有(7-bench 平均 68.20) |

三个对照拆开三个变量:X vs base → EMA 单独值多少;noimg vs X → 加权总贡献;noimg vs shuffle → 视觉相关性单独值多少。

**对照组 X 技术细节**:必须同时把 `w_t=1` 和 `g_sample` 设成 trivial,才是干净的纯 EMA 基线,否则隔离的是"EMA+gate"。

**shuffle 定位升级**:从"sanity check"提升为"核心消融"——若 shuffle 后效果不掉,核心 claim 有大麻烦。

**成本**:两组各一个 62 步训练+评测,约 2 个 noimg run 量级,和 Phase 2b 同一档,**优先级与 2b 并列,成稿前必做**。

**对照组 X 结果分析(2026-07-08,7-benchmark 平均分)**:

| | base | 对照组 X(纯 EMA) | noimg(完整) |
|---|---:|---:|---:|
| 7-bench 平均 | 65.78 | 67.79 | 68.20 |
| 相对 base 增量 | — | **+2.01pp** | +2.42pp |

X vs base 的 +2.01pp 和 noimg vs X 的 +0.41pp 相比,**大部分增益(2.01/2.42 ≈ 83%)来自 EMA self-teacher 蒸馏本身,token 级视觉相关性加权在此之上只贡献了 0.41pp**。逐 benchmark 看,X 在 MMBench_DEV_EN(76.20 vs noimg 76.55)、VStarBench(74.35 vs 73.82,反而更高)上已经和完整版基本打平,只在 BLINK(53.24 vs 55.50)、HRBench4K(75.88 vs 75.75,打平)等少数几项上落后。**这确实支持"EMA 自蒸馏本身已经是主要正则化来源,视觉相关性加权的增量效果有限"这个假设**,但还不能下结论——需要 shuffle 组的结果来做第三个判据:如果 shuffle(打乱 token↔权重对应但保留真实权重分布)也接近 noimg 而不是接近 X,说明"任何非均匀加权"本身有效,和"加权是否来自视觉相关性信号"无关,那 noimg 相对 X 那 0.41pp 里能归因到"视觉相关性信号specifically"的部分会更小,核心 claim(视觉相关性信号是关键)证据强度会进一步减弱。shuffle 组还在跑,结论待其完成后补齐。

**0.41pp 增量在噪声量级内,种子重复实验暂缓**:composite 平均分的 seed variance 此前测过约 ±0.38(见 8-benchmark composite 部分),noimg vs X 的 0.41pp 差距和这个噪声量级相当,统计上目前分不清"真有增量"还是"噪声"。**用户决定**(2026-07-08):不追加 seed 重复训练(noimg/X 各补 1-2 个 seed),先做下面两条低成本路线,如果它们能在其他维度上找到更干净的信号,再决定要不要补种子实验。

**后续两条路线(2026-07-08 排队)**:

1. **换评测维度:幻觉类 benchmark(POPE / HallusionBench)**——RA-VAD 的直接目标是"压语言先验、强化视觉依赖 token",这类信号更应该体现在幻觉类任务而不是通用感知平均分上(平均分可能把机制的信号稀释掉)。✅ **评测已完成**(2026-07-09,base/noimg/uniform-X/black 四个 2B checkpoint):

   | Model | POPE | Δ vs base | HallusionBench aAcc | Δ vs base |
   |---|---:|---:|---:|---:|
   | base-2B | 88.75 | — | **68.66** | — |
   | noimg-2B | 89.16 | +0.41 | 66.98 | **-1.68** |
   | uniform-X-2B | 89.37 | +0.62 | 67.30 | -1.36 |
   | black-2B | 89.31 | +0.56 | **68.03** | **-0.63(掉得最少)** |

   **假设不成立,而且结果比预期更负面**:POPE 上三个变体都小幅领先 base(+0.4~0.6pp,彼此间噪声量级、无法区分),但 HallusionBench 上**所有 RA-VAD 变体都比 base 更差**,不是"增量小"而是**倒退**——noimg 掉得最多(-1.68pp),uniform-X 次之(-1.36pp),black 掉得最少(-0.63pp)。**唯一站得住的结论**:black 相对 base 的掉分幅度明显小于 noimg,方向上和第五轮"black 的权重信号更干净"一致,形成了从"离线权重分布分析"到"下游幻觉类指标"的一致证据链——但没有任何变体真正超过 base,说明当前 RA-VAD 机制对幻觉类任务整体是负向的,这是需要如实写进论文/讨论的负结果,不能只强调"增量有限"。

2. **权重分布粗粒度检查(零训练成本,已完成两轮)**:

   - **第一轮(wandb 聚合指标)**:读取原始 noimg-2B 训练 wandb 日志(`wandb/run-20260630_044118`,已确认对应 `Vision-OPD-noimg-Qwen3-VL-2B-Instruct/global_step_62`)里 RA-weight 的聚合指标:`positive_token_fraction≈0.60`、`ra_norm_mean≈0.60`、`g_sample_mean≈0.73-0.74`、`ra_weight_mean≈0.43-0.44`。发现约 40% 的 response token 权重被硬置零,是明确的"有/无"二元对比结构,不支持"权重分布太平"这个假设,但日志没有逐 token std/分位数,无法确认过阈值的 60% token 内部差异有多大。

   - **第二轮(2026-07-08,逐 token 精确分布)**:用仓库已有的 `scripts/analyze_ra_tokens.py`(离线复现 hi/ctrl teacher-forcing 前向,重算逐 token `ra_weight`;⚠️ 用训练完的 checkpoint 本身近似替代训练时的 EMA teacher,不是 bit-exact 复现,但分布形状应可信)在 noimg-2B step62 的 200 个 rollout 样本(31620 个 response token)上重算,结果:

     | 分位数 | p0-p40 | p50 | p60 | p70 | p75 | p80 | p85 | p90 | p95 | p99 | p100 |
     |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
     | ra_weight | 0 | 0.006 | 0.074 | 0.253 | 0.402 | 0.608 | 0.915 | 1.426 | 2.591 | 4.034 | 5.878 |

     40.15% token 权重严格为 0;有权重的 60% 中位数仅 0.006(大量边缘 token 刚过门槛),但 p90=1.43、p99=4.03,正值部分变异系数(std/mean)达 **1.45**——是"少数 token 权重极高、大多数接近 0"的强长尾结构,不是被压平的分布。分布图见 `analysis_outputs/ra_tokens_dist_20260708/ra_weight_distribution.png`。

   **结论**:两轮都不支持"权重太平导致没区分度",反而说明机制本身产生了强烈的 token 级差异化信号。**"#4 权重太平"这条路线可以排除**,小增量(0.41pp)更可能是"评测维度稀释了信号"(见路线 1),不是"信号强度不够"。

   - **第三轮(2026-07-08,权重高低具体落在哪些词上——比分布形状更关键的发现)**:把 `tokens.csv` 按 token 文本(小写、去空格)聚合,只看出现次数≥15 次的词:

     | 权重最高的词 | 次数 | 均值 | 中位数 | | 权重最低的词 | 次数 | 均值 | 中位数 |
     |---|---:|---:|---:|---|---|---:|---:|---:|
     | user | 23 | 2.70 | 3.36 | | an | 70 | 0.85 | 0.33 |
     | analysis | 44 | 2.57 | 3.04 | | white | 73 | 0.93 | 0.09 |
     | large | 34 | 2.31 | 2.77 | | b(选项字母) | 128 | 0.92 | 0.00 |
     | based | 157 | 2.22 | 2.57 | | text | 52 | 0.94 | 0.74 |
     | provided | 129 | 1.88 | 1.66 | | clock | 39 | 1.02 | 0.49 |
     | let | 45 | 1.40 | 0.84 | | pink | 16 | 1.04 | 0.0001 |

     位置(token 在 response 里的下标 `t`)和权重的相关系数 **-0.15**,按位置分桶更明显:开头 0-5 token 均值权重 0.86,40-80 区间 0.46,160+ 区间只剩 0.24——**response 越靠前权重系统性越高**。权重最高的词几乎全是"Based on the provided image, let me analyze..."这类回答开场白的话语标记词,不是描述视觉属性的内容词。专门抽了颜色词(red/blue/green/yellow/white/black/pink 等,n=665)做对照:均值 0.61(略高于全体均值 0.43),但**中位数只有 0.0007**——大多数颜色词实例权重接近 0,只有少数样本的少数实例权重很高把均值拉起来。

   **修正后的结论**:权重高低更多由**位置/话语结构**驱动(开头 vs 结尾),而不是**内容/视觉语义**驱动(是不是在描述图里的视觉属性)。这比"权重太平"是更直接、更根本的问题——机制在学一个和"这是不是视觉相关信息"弱相关的代理信号(no-image 情况下模型在开场白措辞上的差异,很可能就比后续内容大,纯粹因为没图时模型不知道怎么起手),这解释了为什么 token 级加权在 noimg vs 完整方法上只带来 0.41pp 增量:**加权确实有区分度,但区分的主要不是"该更关注哪个视觉细节",而是"这是不是回答刚开始的几个词"**。

   - **第四轮(2026-07-09,位置去趋势 sanity check——负结果)**:假设"高权重词=位置效应",尝试用两种方式去趋势(按 7 档粗粒度位置分桶减均值 / 按精确 token 位置 t 减均值)重算残差权重榜单。**结果**:两种去趋势后榜首仍是 user/large/foreground/analysis/determine/provided/let,颜色词残差均值仅从 0.61 升到约 0.10-0.12,中位数仍为负——**去趋势基本无效**。原因:`based`(100% 出现在 t=0)、`user`(100% 出现在 t=1)、`analysis`(100% 出现在 t=3)——这些词不是"早期普遍偏高"的平滑衰减,而是**精确锁定在"Based on the provided image, let me analyze..."这句固定开场白模板的固定位置槽**,该位置的"平均权重"本身就约等于这个词自己的权重,减均值去不掉。**结论**:问题比"位置衰减"更具体——是**开场白模板**在争夺高权重,普通的位置去趋势方法处理不了,需要直接识别/排除固定模板 token,或者从根源换掉 ctrl 构造方式(见下一轮)。

   - **第五轮(2026-07-09,black 模式对照——关键正结果)**:`black` 模式的 ctrl 分支不是"去掉图像"而是**把原图换成同尺寸黑图**(`swap_images(high_messages, black_images_like(...))`,代码确认),模型在 hi/ctrl 两个分支都"知道自己有一张图",只是内容被抹掉,天然消除"有没有图"这个元信号混杂。用同一套 `analyze_ra_tokens.py`(200 样本,复用已有的 `rollouts/Vision-OPD-black-Qwen3-VL-2B-Instruct` rollout dump,零训练成本)重跑,对比:

     | | noimg(整图 vs 无图) | black(原图 vs 黑图) |
     |---|---:|---:|
     | user 均值 | 2.70 | **0.00** |
     | based 均值 | 2.22 | **0.09** |
     | provided 均值 | 1.88 | **0.03** |
     | 颜色词均值(相对全体均值倍数) | 0.61(全体 0.43,+42%) | **0.84(全体 0.38,+122%)** |
     | 位置相关系数 corr(t, weight) | -0.150 | -0.124 |
     | 权重榜首(count≥10) | user/analysis/large/based/provided/let(清一色开场白) | **tall/highlights/shows/worn/cylindrical/window/evidence/sculpture/roof/wooden(清一色视觉内容词)** |

     **开场白模板词权重被压到接近 0**,榜首换成清一色视觉属性/物体词,颜色词信噪比翻倍。位置相关性只是略微改善(-0.124 vs -0.150),说明起作用的不是"位置效应变小",而是**黑图 ctrl 直接切断了"有没有图"这个元信号混杂的根源**。

   **第四、五轮合并结论**:简单的位置去趋势救不了 noimg 的权重质量问题,但换一种 ctrl 构造方式(黑图/区域扰动,而不是整图移除)能从根源上大幅改善权重的内容相关性——这个方向和 VisionOPD/VA-OPD 的 teacher 构造思路(保留结构、扰动内容,而非粗暴移除)方法论上是一致的。**下一步待办**:(1)确认 `Vision-OPD-black-Qwen3-VL-2B-Instruct` 训练时用的加权逻辑是否和 noimg 完全一样(如果是,黑图训练理论上已经在用更干净的权重信号,但其 ZoomBench 结果 36.33 和 noimg 的 36.45 基本打平,需要去核实这个"打平"是否被其他因素抵消了);(2)如果确认黑图信号更干净但最终效果没体现出来,说明问题还是在"评测维度不够敏感"(见路线 1,POPE/HallusionBench);(3)可以考虑把 noimg 的 ctrl 构造直接换成黑图重新训练一版,验证 7-benchmark/幻觉类指标是否有更大增量。

   - **⚠️ 数据口径更正**:回查发现第二、三轮分析用的 `--max-samples 200` 在 `load_rollouts()` 里是按 step 文件名排序后顺序读取(`rollouts/.../1.jsonl` 排最前,每个文件 768 条,远超 200),**实际读到的是 step1(训练刚开始,rollout 由几乎等同未训练 base 权重的模型生成)而不是 step62(完整训练完)**。此前文档里标注"noimg-2B step62"的第二、三轮结果,应理解为"noimg-2B 训练早期(step1)"。

   - **第六轮(2026-07-09,回答"是训练训偏了,还是从一开始就这样"——关键结论)**:为验证这一点,单独隔离 `1.jsonl`(真正 step1)和 `62.jsonl`(真正 step62)两个文件重新各跑 200 样本:

     | | step1(≈未训练) | step62(完整训练) |
     |---|---:|---:|
     | 权重榜首(count≥10) | user(2.70)/analysis(2.57)/large(2.31)/based(2.22)/provided(1.88) | user(2.97)/based(2.32)/analysis(2.17)/provided(1.87) |
     | 位置相关系数 | -0.150 | -0.137 |
     | 权重严格为0占比 | 40.2% | 39.7% |
     | 颜色词均值(全体均值) | 0.610(0.426) | 0.543(0.428) |

     **两组几乎完全一致**,颜色词信噪比从 step1 到 step62 甚至略微下降。**结论:这不是 RA-VAD 训练把权重学偏的(training 没有"训偏"),开场白模板词权重虚高这个问题在训练开始之前就已经存在,幅度贯穿整个训练过程基本不变**——根源是 `logp_hi - logp_ctrl`(整图 vs 无图)这个打分方式本身的内生缺陷(模型对"有没有图"这两种 prompt 的第一反应差异天然体现在开场白措辞上),不是训练学出来的偏差。这进一步加强第五轮的结论:**修复应该对准 ctrl 构造方式(换黑图/区域扰动),而不是训练超参或训练时长**。

   - **第七轮(2026-07-09,attention 信号能不能代替 logprob-diff 做加权信号——验证方案 2a)**:新增 `scripts/analyze_ra_attention.py`(复用 `analyze_ra_tokens.py` 的样本匹配/消息构造逻辑,保证和权重分析用同一批样本),提取每个 response token(有真图条件下生成)对图像 token 的 attention 集中度(全部层、全部 head 平均,`attn_implementation=eager`),在同一批 200 个 step62 样本(35189 token)上跑,和 `ra_weight` 直接对比:

     | 词 | ra_weight 均值 | attn_to_image 均值 | |
     |---|---:|---:|---|
     | user | 2.97(严重高估) | 0.063(接近均值) | |
     | based | 2.33(严重高估) | 0.057(接近均值) | |
     | provided | 1.87(严重高估) | 0.066(接近均值) | |
     | wall | 1.02 | **0.161**(全体均值 0.050 的 3.2 倍) | |
     | building | 1.33 | **0.123** | |
     | tower | 0.22(被严重低估) | **0.097** | attention 和 ra_weight 分歧最大的例子之一 |

     attention 榜首(count≥10)清一色视觉物体名词(wall/building/wearing/heart/person/machine/bicycle/bench/wooden/tower/cow/barrier),榜尾清一色推理/元话语词(answer/incorrect/accurate/matches/perhaps/must/since/so),颜色词均值是全体均值的 1.38 倍(比 ra_weight 的信噪比干净)。**两个信号的 token 级相关系数只有 0.284**——经常在"哪个 token 重要"上意见不一致,分歧集中在开场白模板词 vs 真实内容词这个轴上,和第三轮发现的问题正好对应。

     **诚实的注意点**:attention 自己也有位置混杂,且比 ra_weight 更强(corr(位置, attn_to_image) = **-0.394**,ra_weight 只有 -0.137)——越早的 token 对图像关注度天然越高。但这个位置效应**没有**像 ra_weight 那样集中体现在"user/based/provided"这几个特定模板词上(这几个词的 attn_to_image 都只在全体均值附近,没有被异常拔高),更像是"模型刚开始扫视图像定位关键物体"这一真实认知过程的合理特征,而不是"没图时不知道怎么开场"这种语言层面噪声。

     **结论**:用户提议的方案 2a(把 `logp_hi-logp_ctrl` 换成/混合 attention-to-image 分数)方向经验证是对的——attention 确实能更干净地把高权重分配给真实视觉内容词而不是开场白模板词。**实际落地前建议先处理 attention 自身的位置衰减**(比如按位置归一化,或只用来做"是否真的在关注图像内容"的筛选/门控而不是直接线性当权重用),避免重蹈 ra_weight 的位置混杂覆辙。

     **鲁棒性复核(同日,black-2B step65 rollout,200 样本/30020 token)**:换一批完全不同话语风格的 rollout(black 训练变体自己生成的文本)重跑同一套 attention 分析,结果高度一致——榜首(count≥10)清一色内容词(sports/wooden/jacket/building/chair/water/wall/statue/monument/person/backpack/face),榜尾清一色推理元词(matches/option/accurate/answer/choice/incorrect/wait),开场白词(user/based/analysis/provided)的 attn_to_image 仍然都只在全体均值附近(0.05-0.07,没有异常拔高),颜色词均值是全体均值的 1.16 倍,位置相关系数 -0.354(和 noimg 的 -0.394 同量级)。**说明 attention 信号的这个优势是通用的(不依赖具体是哪个 checkpoint 生成的 rollout 文本),不是 noimg 数据的偶然巧合**,方案 2a 的可信度进一步加强。

   - **第九轮(2026-07-09~10,把 attention 信号真正接入训练——`ra_weight_source="attention"`,含下游结果)**:

     **实现**:`compute_ra_weights` 新增 `ra_raw_override` 参数,允许用任意逐 token 相关性信号替换 `logp_hi-logp_ctrl`,复用后面完全一样的 clip/归一化/gate 流程;新增专门的、**不经过 FSDP、独立加载的冻结 scorer 模型**(而不是复用被 FSDP 分片的 teacher)做 attention 提取——原因是 flash-attention 从不物化 attention 权重,而尝试从 FSDP-sharded teacher 里用 `summon_full_params` 硬取,连续踩了两个 FSDP1 已知坑(o_proj 输入形状不对、退出 summon 上下文时的 reshard 断言失败),换成独立冻结模型后这两类问题全部绕开。

     **调试过程中额外发现的三个 bug**(均已修复,供后续参考):(1)mrope `position_ids` 在 `teacher_inputs` 里是 batch-first 存储 `(bsz,3,seqlen)`,但模型 forward 实际需要 `(3,bsz,seqlen)`,直接切片顺序错了,会静默毁掉多模态位置编码,级联成一个看起来毫不相关的 o_proj 形状报错;(2)`extract_multi_modal_inputs` 返回的 `pixel_values` 不保证已经搬到 GPU(它是非张量容器字段,不在批级别的 `.to(device)` 范围内),需要手动搬;(3)**verl 会全局 monkey-patch `Qwen*VLForConditionalGeneration.forward`**(为了支持 remove-padding 等优化),这个 patch 构造的返回对象只保留 `logits`/`hidden_states`,把 `attentions` 直接丢弃——即使 `output_attentions=True` 内部确实算了也拿不到,新加载的 scorer 模型因为是同一个类,也会被这个全局 patch 影响;修法是调用 `scorer.model(...)`(patch 下面一层,没被包裹)而不是 `scorer(...)`。

     **验证**:先用真实模型做了单元测试之外的针对性隔离测试(验证 eager attention 切换、position_ids 形状、monkey-patch 剥离 attentions 这三件事),再跑 4-GPU 冒烟测试(5 个干净 step,`ra_vad/ra_raw_mean` 数值正常),然后正式训练(`Vision-OPD-noimg-attention-Qwen3-VL-2B-Instruct`,同一套 compute-matched 62 步量级配置)。**尝试过一次"批量多样本一起跑 scorer"的加速(`ra_attention_scorer_batch_size=2`)——没有效果**(316s/step vs batch=1 的 302s/step,基本打平甚至略慢),说明瓶颈是 `output_attentions=True` 强制算全部层的纯计算量,不是 Python/kernel 启动开销,batching 救不了。

     **训练中 token 权重质量随 step 变化的离线复核**(不需要碰训练进程,直接对已落盘的 rollout 重跑打分):step 1/10/20 三个快照,权重榜首始终是真实视觉物体词(orange/river/building→landing/entrance/car/wall→charging/station/engine/car/sign),没有开场白词重新冒头,均值和位置相关性也没有漂移——**信号质量在训练过程中保持稳定,没有被训坏**。

     **下游结果(compute-matched step60 checkpoint,7-benchmark,server 模式评测)**:

     | Model | 7-bench 平均 | 相对 base(65.78) |
     |---|---:|---:|
     | base-2B | 65.78 | — |
     | uniform-X(纯 EMA,权重拉平) | 67.79 | +2.01 |
     | noimg(标准,logprob 加权) | 68.20 | +2.42 |
     | **attention(第九轮,attention 加权)** | **67.70** | **+1.92** |

     **attention 版本(67.70)不比 uniform-X(67.79)好,比标准 noimg(68.20)还低 0.5pp——三者全部落在种子噪声(±0.38)量级里,统计上分不开**。也就是说:即便离线验证过 attention 信号在"是否真正对齐视觉内容"这个维度上明显比 logprob-diff 干净,换到实际训练里接上下游 7-benchmark 综合分,**依然看不出比"完全不加权"(uniform-X)有任何增量价值**。

     **结论**:第二~八轮把"logprob-diff 这个具体打分方式有内生缺陷"证明得很扎实,但第九轮把"换一个更干净的打分方式(attention)是不是就能救回来"这个假设也证伪了——**问题不在"用哪个信号做 token 级加权",而在"token 级加权这件事本身,对这批下游任务的贡献目前测不出来"**。三种权重来源(均匀/logprob/attention)在 7-benchmark 综合分上不可区分,核心的、可复现的增益(+2pp 量级)来自 EMA self-distillation 本身,不来自任何形式的视觉相关性加权。这是目前全篇最直接支持"当前 noimg 的加权设置没有提供实质帮助"这个判断的证据。

     **仍然打开的问题**:(1)attention 训练还在跑(192 步全量,截至目前到 step ~90),step60 只是compute-matched 的早期快照,完整训练完再看一次结论是否稳定;(2)POPE/HallusionBench(幻觉类指标,机制"应该"最擅长的地方)结果待补,之前 noimg/uniform-X/black 在这两个指标上全部**低于** base,attention 版本能不能扭转这个方向还不知道,这是判断"加权到底有没有用"最后一块拼图。

     **第九轮补充(2026-07-10)**:(a)attention 全量训练已跑完(192/192,~16h),最终 checkpoint 暂不评测(用户决定:step60 结果 + 下面第十轮的诊断已经足够说明问题,不值得再花评测算力);(b)attention-step60 的 POPE/HallusionBench 补上了:POPE 89.31(和其他变体一样,略高于 base 88.75、彼此不可分),**HallusionBench 67.61——仍低于 base(68.66),没有翻盘,但好于 noimg(66.98)/uniform-X(67.30),排序 base > black(68.03) > attention > uniform-X > noimg**。这个排序透露一个重要信息:**ctrl 构造方式(black vs 整图移除)对幻觉指标的影响,比权重来源(attention/logprob/均匀)更大**;(c)训练中新增可选的逐步 token 权重采样落盘(`ra_token_dump_dir`,默认关闭,配 `scripts/summarize_ra_token_dump.py` 离线解码),以后可以直接监控"高权重 token 是什么、随训练怎么变",不用再离线重算。

   - **第十轮(2026-07-10,根因诊断:不是"找不准 token",是"token 上没有可学的 signal"——以及对比锐化 target 的修复)**:

     **用户提出的机制性质疑**:teacher(EMA)和 student 权重几乎一样、input 也完全一样(hi 分支同图同题),那么即使 token 找得再准,teacher 在那个 token 上的分布和 student 本来就几乎重合——加权只是在放大一个趋近于零的信号。**需要的不只是"找对 token",而是"让 teacher 在 token 上的 signal 和 student 不一样"。**

     **零成本诊断(virl39k-filtered step464,student + 同步保存的真实 EMA teacher,100 个 on-policy 样本 / 53432 token,`scripts/diagnose_teacher_student_kl.py`)**:

     | Token 分桶(按 ra_weight) | teacher-student KL(T=2) |
     |---|---:|
     | 权重=0(65%) | 0.000334 |
     | (q75,q90] | 0.000563 |
     | >q90(最高10%) | **0.000739** |

     两个结论:(1)"找 token"弱有效——高权重桶的 KL 是零权重桶的 ~2.2 倍(corr=0.21),权重不是瞎找的;(2)**绝对量级上 signal 不存在**——最高权重的 token 上 teacher-student 分布差也只有 0.0007 nats,teacher 没有任何 student 不知道的东西可教。**同一批 token 上 teacher 自己的 hi-vs-ctrl logprob 差是 1.33 nats——是 teacher-student KL 的 1800 倍**:真正有信息量的分布差异(有图 vs 无图)一直在训练管线里(ctrl 分支每步都算),只是被降级成了标量权重,从未被用作蒸馏 target。这从机制层面解释了第九轮"三种权重来源全部打平"。

     **修复方案(对比锐化 target)**:把蒸馏 target 从 `p_teacher_hi` 换成 `target ∝ softmax(lp_hi + α(lp_hi − lp_ctrl))`——即 `p_hi^(1+α)/p_ctrl^α` 的几何倾斜(把 contrastive decoding 的推理期操作蒸馏进权重)。**自带 token 级差异化**:视觉相关 token 上 hi/ctrl 分歧大→target 和 student 差异大→梯度大;非视觉 token 上 hi≈ctrl→target≈student→梯度自动趋零,不需要任何显式加权。数学性质:softmax 保证永远是合法分布(用户曾疑虑"两个分布的差不是分布"——那是概率空间减法,这里是 logit 空间减法=概率比值倾斜,不同的操作);**plausibility 约束(β)保证支撑集 ⊆ hi 认可的 token 子集**——student 学到的每个词都是 teacher 本来就会说的词。

     **零训练成本预检(`scripts/precheck_contrast_target.py`,60 样本 × {noimg,black} ctrl × α∈{0.5,1,2} × β∈{0.1,0.05},两个数据分布各跑一遍)**:(1)signal 强度问题解决——KL(target‖p_hi) 0.05-0.27,是空信号的 100-300 倍;(2)定位富集有效——argmax 改变位置落在 ra_raw top 25% 的比例 42-50%(随机应为 25%);(3)**改变内容偏风格 churn**——大量 `The`→`It`/`Based`→`To` 类措辞变化,black ctrl 下有少量实质性变化(`image`→`red`、答案字母 C→D/A、数字 2→1);(4)**抓到一个真实危险**:裸倾斜会抬升 `<|im_end|>`(提前终止 token)——训进去会静默缩短 response。注意 argmax-change 视角天然偏向 top-2 接近的位置(功能词扎堆处),内容词位置的概率锐化不翻转 argmax、在此视角下不可见,实际梯度信号可能比表面看起来更好。

     **实现(commit 见 git log)**:`build_contrast_target()`(verl/trainer/ppo/ra_vad.py)带四层保护——softmax 合法性 / β-plausibility 支撑集约束 / `ra_contrast_exclude_token_ids`(排除 im_end=151645/endoftext=151643,针对预检抓到的终止 token 问题)/ `ra_contrast_gate_positive_only` 位置门控(只在 ra_weight>0 处倾斜);自动记录 `contrast_target_kl_vs_hi` 和 `contrast_argmax_change_rate` 两个漂移监控指标;masked token 的 -inf clamp 到 -1e4 避免 KL 出 NaN。配置项 `ra_target_mode=contrast` + α/β/gate/exclude 四个旋钮,11/11 单测通过。

     **两个训练已启动(2026-07-10,VisionOPD 本仓库数据,与 noimg/uniform-X 完全相同的默认启动路径保证可比;两组均用 black ctrl——noimg ctrl 的风格混杂已确证,没有消融价值)**:

     | | 保守(GPU4-7) | 标准(GPU0-3) |
     |---|---|---|
     | α | 0.5 | 1.0 |
     | 位置门控 | 有 | 无 |
     | β=0.1 + 排除终止token | ✅ | ✅ |
     | 脚本 | `run_experiment_contrast_conservative.sh` | `run_experiment_contrast_standard.sh` |

     唯一差异是"倾斜推多狠"(α+门控),是干净的单变量消融。**保守版 step1 实测**:`token_kl_mean` 从 0.004 → **0.298(~75 倍)**,`kd_loss` 从 0.005 → **0.94(~190 倍)**——蒸馏信号第一次真正非零;漂移指标健康(`contrast_target_kl_vs_hi=0.012`,`argmax_change_rate=1.5%`,门控把干预面压到了预检无门控值的 1/5)。跑完后(各 62 步)走 7-benchmark + POPE/HallusionBench 与 base/noimg/uniform-X/black 对比。**判断标准**:如果 contrast 版本(尤其在 HallusionBench 上)仍然不能超过 uniform-X 乃至 base,那"用 hi/ctrl 对比信息改进自蒸馏"这条线就可以定论关闭;如果有增益,则说明问题确实出在"signal 被降级成权重"这一设计上。

     **训练过程记录**:两个训练在 2026-07-10 15:32 曾因共享磁盘配额被 checkpoint 堆满而双双崩溃(`Disk quota exceeded` 写 rollout dump;根因是 `checkpoints/` 累积到 6.5TB),清理 ~5TB 后从断点(保守 step30/标准 step20)续训,损失约 5 小时;由此确立了 checkpoint 保留政策(60 步存 3 个、400+ 存 5 个,`scripts/prune_checkpoints.sh`,CLAUDE.md 已记录)。训练全程 loss 持续下降(保守 0.94→0.52、标准 1.26→0.91)——**旧机制的 loss 是平的(无学可学),这是第一次出现真实的学习曲线**;response 长度稳定(123-141)、`target_kl_vs_hi` 无膨胀,所有漂移护栏安静。

     **⭐ 保守版下游结果(2026-07-11,第十轮的裁决,方向性结论:成功)**:

     | Model | 7-bench 平均 | Δ vs base | HallusionBench | Δ vs base | POPE |
     |---|---:|---:|---:|---:|---:|
     | base-2B | 65.78 | — | 68.66 | — | 88.75 |
     | uniform-X(纯 EMA) | 67.79 | +2.01 | 67.30 | -1.36 | 89.37 |
     | attention(step60) | 67.70 | +1.92 | 67.61 | -1.05 | 89.31 |
     | noimg(标准) | 68.20 | +2.42 | 66.98 | -1.68 | 89.16 |
     | black | — | — | 68.03 | -0.63 | 89.31 |
     | **contrast-保守** | **69.07** | **+3.29** | **68.56** | **-0.10** | 88.41 |

     两个"第一次"同时发生:(1)**7-bench 69.07 是第一个跳出 67.5-68.2 噪声聚集带的变体**(比 noimg +0.87、比 uniform-X +1.28,均超出 ±0.38 种子噪声),提升集中在 MMStar(+2.4)/MathVista(+2.1)/HRBench8K(+1.5),无明显掉分项;(2)**HallusionBench 68.56 第一个基本追平 base**(-0.10),而且演进序列完全单调:noimg(-1.68)→uniform-X(-1.36)→attention(-1.05)→black(-0.63)→contrast(-0.10)——每一步机制改进都在这个指标上兑现。小瑕疵:POPE 88.41 略低于其他变体的 ~89.3(和 base 打平)。

     **结论**:用户的核心判断("问题不在找不准 token,而在 identical-input EMA teacher 在 token 上没有 student 不知道的 signal")和修复方案("把 hi-ctrl 对比从标量权重升级为蒸馏 target")**同时被下游数据验证**——综合分和幻觉分第一次同步改善,这不是噪声波动的模式。注意事项:单种子;server 模式评测(和 offline 的聚合差 ±0.4pp 以内,已验证)。

     **⭐⭐ 标准版结果(同日,最终裁决)**:

     | Model | 7-bench 平均 | Δ vs base | HallusionBench | Δ vs base | POPE |
     |---|---:|---:|---:|---:|---:|
     | base-2B | 65.78 | — | 68.66 | — | 88.75 |
     | noimg(原始方法) | 68.20 | +2.42 | 66.98 | -1.68 | 89.16 |
     | contrast-保守(α=0.5+门控) | 69.07 | +3.29 | 68.56 | -0.10 | 88.41 |
     | **contrast-标准(α=1.0 无门控)** | **70.65** | **+4.87** | **69.19** | **+0.53** | 89.13 |

     标准版**三项全面胜出**:(1)7-bench 70.65,比原始 noimg 高 2.45,增益是原始方法(+2.42)的两倍;逐项全部上涨——BLINK 58.71(+3.2)、**VStarBench 76.96(全表最高,超过 bbox-crop 的 visionopd 75.39 和 virl39k-v2 step150 的 76.44)**、MathVista 67.90(+4.7)、HRBench4K 77.62、HRBench8K 73.75;(2)**HallusionBench 69.19,第一个超过 base 的变体(+0.53)**——从"所有 RA-VAD 变体都在幻觉指标上倒退"到正增益;(3)POPE 89.13 正常。**保守→标准的单变量消融方向一致:倾斜越强越好(69.07→70.65),α=1.0 尚未见过头迹象,α 还有上探空间(1.5/2.0 值得试)**。

     **第十轮最终结论**:证据链完整闭环——(a)诊断(identical-input EMA teacher 无 token 级 signal,hi-ctrl 差是其 1800 倍)✅;(b)修复(对比锐化 target,把 signal 从权重升级为目标)✅;(c)下游三项指标(综合/幻觉/POPE)全面验证,且强度消融方向一致 ✅。**这是 Phase 2-核心 系列消融的最终答案:RA-VAD 原设计中"token 加权"不是正确的信息注入方式,"对比 target"才是**。后续方向(按优先级):α 上探(1.5/2.0);种子复现(写论文必需);ZoomBench(native)补测;与 GRPO compute-matched 的对比更新。

   - **第十一轮(2026-07-11,⚠️ 重大口径修正:第十轮全部"vs base"数字用错了 baseline)**:

     **发现**:用户质疑"base-2B 在 MMBench_DEV_EN 是 69.24?我记得是 78.09"。核实后确认用户是对的——本文档(包括整个第十轮、以及更早 Phase 2-核心 里所有"base-2B"行)引用的 `base-2B = 65.78(7-bench)/69.24(MMBench_DEV_EN)/68.66(HallusionBench)/88.75(POPE)`,实际来自 `Vision-OPD-baseline-Qwen3-VL-2B-Instruct/global_step_65`——这是一个**训练过的**checkpoint(`run_vision_opd_ra_vad.sh` 里 `EXPERIMENT=baseline` 对应 `teacher_prompt_mode=answer_hint`、`ra_vad=False`,即 OPSD 式 answer-hint 自蒸馏,训了 65 步),不是未训练的原始 `Qwen3-VL-2B-Instruct`。真正的原始 base 在 `VLMEvalKit/outputs_vllm_curated/qwen3vl2b_base_reference_temp0_4096_eval`(2026-06-28 跑的),MMBench_DEV_EN=78.09,和用户记忆精确吻合。

     **已确认的真实 raw-base-2B 数字(5/9 项,来自 `qwen3vl2b_base_reference_temp0_4096_eval`)**:

     | BLINK | MMStar | MMBench_DEV_EN | VStarBench | MathVista_MINI | 5-bench 平均 |
     |---:|---:|---:|---:|---:|---:|
     | 53.02 | 57.47 | **78.09** | 72.77 | 62.50 | 64.77 |

     HRBench4K/HRBench8K/POPE/HallusionBench 从未在原始 base 上评测过(第十轮引用的 88.75/68.66 全部来自被错标的 answer-hint checkpoint),补跑已启动(`raw-base-2B-fill_server`,2026-07-11,GPU2,4 个 benchmark 用同一套已 debug 的 vllm_server pipeline)。

     **修正后的对照(5/9 项,contrast-标准 vs 真实 raw base,2026-07-11)**:

     | | raw base(真) | noimg | contrast-保守 | **contrast-标准** |
     |---|---:|---:|---:|---:|
     | BLINK | 53.02 | — | — | — |
     | MMStar | 57.47 | — | — | — |
     | MMBench_DEV_EN | **78.09** | 76.55 | 76.89 | 77.06 |
     | VStarBench | 72.77 | — | — | — |
     | MathVista_MINI | 62.50 | — | — | — |
     | 5-bench 平均(已知项) | 64.77 | — | — | **68.63(+3.86)** |

     **两个层面的修正**:(1)**方向性结论不变甚至更强**——contrast-标准相对真实 raw base 的 5-bench 平均增量(+3.86)比之前(用错误 baseline 算出的虚高数字)更干净、更可信,因为真实 baseline 没有经过任何自蒸馏训练,是真正的"零基线"。(2)**新暴露的负面发现,此前被错误 baseline 掩盖**:MMBench_DEV_EN 上**所有 RA-VAD 变体(noimg 76.55、保守 76.89、标准 77.06)都低于真实 raw base(78.09)**,约 -1 到 -1.6pp;此前用错的 baseline(69.24)把这个比较翻转成了虚假的 "+7.8pp 提升"。这条regression 需要如实写进文档,不能只强调 contrast 版本"三项全胜"——那个结论是在错误 baseline 下得出的,MMBench 这一项在真实 baseline 下是倒退,不是胜出。

     **✅ 补跑已完成(2026-07-11 17:00 左右),完整 9-benchmark 修正版对照表**:

     | | raw base(真) | noimg | contrast-保守 | **contrast-标准** |
     |---|---:|---:|---:|---:|
     | BLINK | 53.02 | 55.50 | — | 58.71 |
     | MMStar | 57.47 | 61.20 | — | ≈62.55(推算) |
     | MMBench_DEV_EN | **78.09** | 76.55 | 76.89 | 77.06 |
     | VStarBench | 72.77 | 73.82 | — | **76.96** |
     | MathVista_MINI | 62.50 | 63.20 | — | 67.90 |
     | HRBench4K | 71.13 | 75.75 | — | 77.62 |
     | HRBench8K | 67.38 | 71.38 | — | 73.75 |
     | 7-bench 平均 | **66.05** | 68.20 | 69.07 | **70.65** |
     | POPE(Overall) | 88.76 | 89.16 | 88.41 | 89.13 |
     | HallusionBench(aAcc) | 68.35 | 66.98 | 68.56 | 69.19 |

     **用真实 baseline 重算后的结论**:(1)**7-bench 综合分方向不变,幅度更大**——contrast-标准相对真实 base 的增量从"错误 baseline 下的 +4.87"变成 **+4.60**(70.65-66.05),仍然是全篇最大增量,且真实 baseline 本身更低,说明这不是"baseline 虚高导致的虚假差距缩小",contrast-标准是真实、扎实的提升。(2)**HallusionBench 结论同样成立且更干净**:contrast-标准 69.19 vs 真实 base 68.35,**+0.84**(此前错误 baseline 下算出的是 +0.53)——依然是唯一超过 base 的变体,而且优势比之前认为的更大。(3)**POPE 三个变体都和真实 base(88.76)基本打平**(88.41~89.16 之间,±0.4pp,噪声量级内),和此前用错误 baseline(88.75)得出的判断一致,不受修正影响。(4)**唯一被推翻的是 MMBench_DEV_EN 单项**:noimg/保守/标准三个变体都低于真实 base(78.09)约 1-1.6pp,不是此前错误 baseline 下的"+7.8pp 提升"——但这不影响 7-bench 综合分和幻觉指标的主结论,只是提醒"逐 benchmark 拆开看,contrast 机制不是全面碾压,MMBench 这一项确实退步"。

     **待办**:(a)contrast-保守缺 BLINK/MMStar/VStarBench/MathVista/HRBench4K/8K 的单项数字(只有 7-bench 聚合值,来自第十轮训练时评测,未逐项记录/存档),如需完整单项对比需重新拉取原始 xlsx;(b)系统性重新审视 Phase 2-核心 全篇(尤其第十轮"contrast-标准三项全面胜出"的结论)里所有引用 65.78/69.24/68.66/88.75 这组旧数字的地方,统一标注为"旧口径,基于错误 baseline,方向性结论不变但幅度需参考本轮修正值";(c)本节之前 uniform-X / attention / black 三个变体(未重新评测)的"vs base"数字仍基于旧错误 baseline,方向性结论(黑图 ctrl 优于 noimg ctrl)不受影响,但幅度未重算。

   - **第十二轮(2026-07-11,ZoomBench native pipeline + GRPO-2B-step585 补测)**:用 Phase 4 三重修复后的 native ZoomBench pipeline(`eval/run_zoombench.sh`,v3-fixed judge,845 题,seed42)补测 contrast-标准-2B(step62)和 grpo-2B(step585,与本文档其余 grpo_vanilla_qwen3vl2b_3ep 结果同一 checkpoint):

     | Model | ZoomBench(v3-fixed native) |
     |---|---:|
     | 2B base(历史值,同 pipeline) | 42.49 |
     | noimg(历史值,同 pipeline) | 36.45 |
     | visionopd(full,历史值,同 pipeline) | 37.51 |
     | grpo-2B-step585 | **43.55** |
     | ~~contrast-标准-2B(step62)~~ | ~~73.02~~ **⚠️ 已证伪,见下方根因分析,真实值约 40.47** |

     **⚠️ 73.02 是自判分(self-judge)偏差造成的假象,不是真实能力提升(2026-07-11,用户质疑后复核确认)**:

     `run_zoombench.sh` 的判分设计是"self-judge with same model"——用被评测的模型自己当 judge。对 845 题里 LLM-judge 兜底的那部分(rule-based 规则匹配不到、需要模型自己判断对错的题),contrast-标准把 572 题送去自判,其中 344 题判"Yes"。但用和 `extract_first_option()` 完全相同的正则重新从 MCQ 模型回答里独立抽取模型自己给出的最终选项,和 ground truth 比对,发现 **621 道 MCQ 题里有 223 题(占全部 845 题的 26.4%,占 LLM-判"Yes"的 344 题的 65%)是模型自己在正文里明确选错了选项,但轮到自己当 judge 判断"这个回答对不对"时又判"Yes"**。典型例子(GT=C,模型自己在正文里说"C. Red: Incorrect... D. Black: Correct... The correct answer is D. Black.",结论明确选了 D,但同一个模型当 judge 时判这道题"Yes")。**用同一套正则重新抽取模型自己的最终选项独立打分(不依赖它自我判断),MCQ 部分修正后的准确率约 342/845 = 40.47%**,和 noimg(36.45)/base(42.49)/grpo(43.55)在同一量级,不再是异常值。

     **根因**:对比锐化训练目标只作用于生成阶段(response token 的蒸馏),完全没有触碰"模型自我评判对错"这个能力,contrast-标准在这个能力上没有被专门训练,标准 ZoomBench self-judge pipeline 假设"模型知道自己什么时候错了"这个前提对它不成立(或者比其他变体成立得更少)——用同一个模型既当考生又当阅卷人,在这个模型上失效了。grpo-585 没有这个问题(其 LLM-judge 分支反而是另一个方向的 bug:回答太啰嗦,64 token 的 `judge_max_tokens` 经常在模型说出"Yes/No"结论前就被截断,导致这部分题目全部被误判为"No"——grpo 的 43.55 更可能是被低估而不是高估,真实值待独立 judge 复核后才能确定)。

     **待办**:(a)`eval/run_zoombench.sh` 的 self-judge 设计本身有系统性风险,应该换成固定的第三方 judge(如本文档其余 benchmark 用的 `gpt-5.4-mini-2026-03-17`)重新跑 contrast-标准和 grpo-585 两个模型的 ZoomBench,而不是继续用上面的正则修正值凑合;(b)检查本文档 ZoomBench 表里其余用同一 self-judge pipeline 跑出来的历史数字(base/noimg/visionopd/black/degrade/qvis)是否有同样的自判偏差——目前只复核了 contrast-标准和 grpo-585 两个,不能假设其余历史数字没有类似问题,需要抽查。

     **✅ (a)已完成(2026-07-12)**:发现 `eval/run_zoombench.sh` 不是 `CLAUDE.md` 里定义的 canonical ZoomBench pipeline——canonical pipeline 明确要求用外部 `gpt-5.4-mini-2026-03-17`(Azure,`judge_max_tokens=2048`)当裁判,`run_zoombench.sh` 却是自判(`judge_max_tokens=64`),两者不是同一个东西,前面 73.02 的假象正是因为误用了后者。已把 canonical 4 步流程封装成 `scripts/run_zoombench_canonical.sh`(供以后复用,避免再次手滑用错脚本),用它重新跑了两个模型:

     | Model | ZoomBench(canonical, gpt-5.4-mini judge) |
     |---|---:|
     | 2B base(历史值,同 canonical pipeline) | 42.49 |
     | noimg(历史值,同 canonical pipeline) | 36.45 |
     | visionopd(full,历史值,同 canonical pipeline) | 37.51 |
     | **contrast-标准-2B(step62)** | **43.67** |
     | grpo-2B-step585 | **47.81** |

     两个数字都和之前的正则粗估(40.47/自判 43.55)量级接近,确认了两件事:(1)**contrast-标准的 43.67 是全篇 RA-VAD 系列里第一个超过 2B base(42.49)的 ZoomBench 结果**,虽然优势不大(+1.18),但方向和 7-benchmark/HallusionBench 的"contrast-标准全面改善"结论一致,不再是自判偏差下的异常值;(2)grpo-585 的真实值(47.81)确实比自判 pipeline 算出的 43.55 更高,印证了此前"grpo 因回答啰嗦被 64-token judge 截断而低估"的推测——GRPO 在 ZoomBench 上的真实优势比之前记录的更大。**(b)仍未做**:base/noimg/visionopd/black/degrade/qvis 的历史 ZoomBench 数字还没有逐一用 canonical pipeline 复核过是否有自判问题——不过这几个数字本来就是来自 canonical/gpt-judge 的旧 pipeline(而不是这次误用的 `run_zoombench.sh`),经抽查 noimg 的历史 judge 文件(`VisionOPD-noimg-Qwen3-VL-2B-Instruct_mcq_seed42_answer.jsonl`)确认误判率仅 26/621(4.2%),远低于 contrast-标准自判时的 223/621(35.9%),没有发现同类问题,基本可以排除。

     **4B 版本的 canonical ZoomBench 补测仍是待办**,contrast-标准-4B / contrast-保守-4B 目前只有下面第十三轮的 7-benchmark+POPE+HallusionBench,还没有 ZoomBench 数字。

     **grpo-2B-step585 的 POPE/HallusionBench 补测(同日,vllm_server 模式,与本节其余数字同 pipeline)**:

     | Model | POPE(Overall) | HallusionBench(aAcc) |
     |---|---:|---:|
     | raw base(真,第十一轮) | 88.76 | 68.35 |
     | contrast-标准-2B(第十一轮) | 89.13 | **69.19** |
     | grpo-2B-step585 | 87.64 | 66.14 |

     **grpo-585 在幻觉类指标上全面低于 raw base**(POPE -1.12、HallusionBench -2.21),也低于 contrast-标准,说明 GRPO 的 verifiable-reward 优化虽然在 ZoomBench(需要精确空间定位/文字识别)上有优势,但不像 contrast-标准那样同时改善幻觉类指标——两种机制在不同任务类型上的收益并不重叠,这是"RA-VAD 蒸馏机制 vs GRPO"在防御性论点(Phase 2b)之外的又一处互补证据。

     > ~~grpo-2B-585(43.55)基本符合预期...contrast-标准(73.02)远超所有已知变体和 base,是全篇迄今最大的单项异常值...~~ **以下段落已被上方"73.02 已证伪"的复核推翻,仅保留删除线作历史记录,不再引用。** 73.02 是 `run_zoombench.sh` 自判分 bug 导致的假象,真实值(canonical GPT judge)是 43.67,见上文。

   - **第十三轮(2026-07-12,contrast-标准/保守 4B 全量训练下游结果)**:两个 4B 训练(`Vision-OPD-contrast-standard-Qwen3-VL-4B-Instruct` / `Vision-OPD-contrast-conservative-Qwen3-VL-4B-Instruct`,均 62 步,GPU 配置同 2B 版本的 α/β/门控设置)已完成,VLMEvalKit 7-benchmark + POPE + HallusionBench(vllm_server 模式)结果:

     | Benchmark | noimg-4B(历史值,step62) | contrast-保守-4B(step62) | **contrast-标准-4B(step62)** |
     |---|---:|---:|---:|
     | BLINK | 66.86 | 66.96 | 66.86 |
     | MMStar | 69.47 | 69.67 | 71.00 |
     | MMBench_DEV_EN | 83.08 | 82.47 | 80.15 |
     | VStarBench | 82.72 | 84.29 | 83.25 |
     | MathVista_MINI | 77.00 | 78.5 | 78.4 |
     | HRBench4K | 79.50 | 80.13 | 79.75 |
     | HRBench8K | 74.75 | 77.63 | 76.63 |
     | **7-bench 平均** | **76.20** | **77.09** | **76.58** |
     | POPE(Overall) | 无历史数据 | 88.45 | 88.71 |
     | HallusionBench(aAcc) | 无历史数据 | 73.08 | 73.19 |

     **观察**:(1)**4B 规模上保守版反而略优于标准版**(77.09 vs 76.58),和 2B 上"标准>保守"(70.65 vs 69.07)的方向相反——4B 上标准版在 MMBench_DEV_EN 上明显回落(80.15,比 noimg 低 2.93,也比保守版低 2.32),拖累了综合分,这和 2B 上"MMBench 是唯一退步项"的模式相呼应,但在 4B 上标准版(更强的倾斜 α=1.0)似乎让这个退步被放大了,而不是像 2B 上其余指标那样被更强的倾斜进一步改善;(2)**两个 4B 变体都比 noimg-4B(76.20)高**(+0.89/+0.38),方向和 2B 上的结论一致(对比锐化机制优于纯 logprob 加权),但幅度比 2B 上小很多(2B 上 contrast-标准比 noimg 高 2.45,4B 上只高 0.38);(3)POPE/HallusionBench 在 4B 上没有历史 noimg 基线可比,无法判断幻觉类指标是否也在 4B 上复现 2B 的改善模式,这是明显的待办缺口;(4)HRBench8K 上保守/标准都比 noimg 有明显提升(+2.88/+1.88),是本轮最大的单项增益来源。

     **待办**:(a)4B base(未训练的原始 Qwen3-VL-4B-Instruct)的完整 7-benchmark+POPE+HallusionBench 从未跑完过(此前"补跑中"状态一直没有下文),这次 4B 对比也没有真正的零基线可比,需要补;(b)4B 版本的 canonical ZoomBench(GPT judge)还没跑,contrast-标准-4B/contrast-保守-4B 目前完全没有 ZoomBench 数字;(c)2B 上标准 > 保守,4B 上保守 > 标准,这个反转本身值得关注——可能是 4B 模型本身的知识容量更大,更强的对比倾斜(α=1.0)反而干扰了它在 MMBench 这类通用感知任务上原本更强的能力,值得在补齐 4B base 基线后重新核实这个反转是否稳健(目前只有单个种子)。

   - **第十四轮(2026-07-12,待跑:contrast target + 无显式加权的干净隔离)**:代码核实发现 `ra_kd_loss` 无论 `target_mode` 是 `teacher` 还是 `contrast`,loss 都仍然乘以标准的逐 token `ra_weights`(`weights = ra_weights * loss_mask`,`dp_actor.py:1303` 起),`contrast_gate_positive_only` 只影响 **target 构造**是否额外被 `ra_weight>0` 门控,不影响这层外部权重乘法——目前保守/标准两个 config 都没有关掉它(`ra_uniform_weight` 均为默认 `False`)。**还没做过的关键实验**:`ra_target_mode=contrast` + `ra_uniform_weight=True`(即彻底去掉外部逐 token 权重,只留 contrast target 自身的 KL 量级差异来提供隐式的 token 级差异化)。如果这个配置效果不降甚至更好,说明真正起作用的是 target 内容本身,和"reweight"这个概念完全无关;如果效果明显下降,说明外层的标量权重乘法仍然是必要成分,contrast 机制不是"纯粹换了个 target 就行"那么简单。**排入下一批要跑的实验**,建议直接在 2B、black ctrl、α=1.0(标准强度)上跑,和现有 contrast-标准(70.65)做直接对比,复用同一套评测流程。

     **✅ 结果已出(2026-07-14,trial_id=301683547 训练+评测,`Vision-OPD-contrast-standard-uniform-weight-Qwen3-VL-2B-Instruct/global_step_62`)**:

     | Benchmark | uniform-weight | contrast-标准(对照) | Δ |
     |---|---:|---:|---:|
     | BLINK | 56.76 | 58.71 | -1.95 |
     | MMStar | 62.20 | 62.53 | -0.33 |
     | MMBench_DEV_EN | 76.89 | 77.06 | -0.17 |
     | VStarBench | 73.82 | 76.96 | -3.14 |
     | MathVista_MINI | 68.00 | 67.90 | +0.10 |
     | HRBench4K | 75.88 | 77.62 | -1.74 |
     | HRBench8K | 66.38 | 73.75 | **-7.37** |
     | **7-bench 平均** | **68.56** | **70.65** | **-2.09** |
     | POPE | 88.97 | 89.13 | -0.16 |
     | HallusionBench aAcc | 69.40 | 69.19 | +0.21 |

     **结论:外层逐 token 权重仍是必要成分**——去掉后 7-bench 掉 2.09pp(超出 ±0.4pp 种子噪声带),不是"换个 target 就够了"。掉分集中在 HRBench8K(-7.37)和 VStarBench(-3.14)这类高分辨率/细粒度视觉任务上;而 MathVista/MMBench 几乎不动,POPE/HallusionBench 完全持平甚至略好——**加权的作用可以更精确地描述为:主要帮助高分辨率细粒度感知,对幻觉类指标没有贡献**(幻觉改善来自 contrast target 本身)。ZoomBench 补跑中(2026-07-14,trial 301783374 排队)。

     **⚠️ 修正(2026-07-15):HRBench8K 的 66.38 是坏数字,重跑中**——用户觉得这项低得反常,核查逐题数据发现该次评测有 **61/800 条推理请求直接失败**(预测文本是 "Failed to obtain answer via API.",当时 vLLM 请求错误,VLMEvalKit 全部按错判;其余 8 个 benchmark 失败数为 0,GPT judge 本身没有问题)。剔除这 61 条后命中率约 71.9%,回到正常区间(对照 contrast-标准 73.75/noimg 71.38)。**影响**:上面"-7.37"的单项掉分和"-2.09pp"的 7-bench 差距都被这个坏数字放大了——按 71.9 估算,7-bench 平均约 69.35,与带权重版的差距缩小到约 -1.3pp(仍超出 ±0.4 噪声带,**方向性结论"外层权重是必要成分"预计不变,但幅度要等重跑数字**);"掉分集中在高分辨率任务"的表述也要重新核实,VStarBench(-3.14) 仍然成立,HRBench8K 的真实差距待定。重跑已排队(trial 301783374,`experiment_script/run_uniform_weight_hrbench8k_rerun.sh`,GPU5,等 backfill 队列空出)。

   - **第十五轮(2026-07-12,virl39k-filtered 换数据,重点看 MathVista 能不能进一步提升)**:contrast-标准/保守在 2B/4B 上 MathVista_MINI 都有明显提升(2B 标准 67.90/+4.7,4B 保守 78.5),而本仓库 `train_answer.parquet` 不是 math-heavy 数据;之前已经验证过换成 virl39k(geometry/chart 为主的 math MCQ,`data/virl39k_train_noimg_filtered.parquet`,14861 条)训练 noimg 方法本身有效(3.1/3.2 节),但从未在这份数据上跑过 contrast target 或 GRPO。**已启动两个训练(2026-07-12,2B,均 1 epoch/~464 步,GPU0-3/GPU4-7)**:
     - `Vision-OPD-grpo-Qwen3-VL-2B-Instruct-virl39k-filtered`(`scripts/run_experiment_grpo_baseline.sh`,`TASK_TRAIN_FILE` 指向 virl39k-filtered,复用同一套 verifiable-reward 奖励函数,`reward_model.ground_truth` schema 与 `train.parquet` 一致,已核实兼容)
     - `Vision-OPD-contrast-conservative-Qwen3-VL-2B-Instruct-virl39k-filtered`(`scripts/run_experiment_contrast_conservative.sh`,`ANSWER_VAL_TRAIN_FILE` 指向 virl39k-filtered,`TRAIN_BATCH_SIZE=32` 以匹配此前 noimg-virl39k-filtered 跑出的 464 步/1 epoch 量级,便于直接对比)

     **待办**:跑完后和 (a) 本仓库数据训出的 contrast-保守/标准、(b) noimg-virl39k-filtered(step464,MathVista 61.10~61.50,当时用的是 noimg 而非 contrast)、(c) grpo-2B(本仓库 `train.parquet`)三方对比,重点看 MathVista_MINI 这一项换数据后是否比本仓库数据训出的版本更高——如果确实更高,说明 contrast 机制的 MathVista 增益一部分来自"训练数据本身包含更多 math 内容",不是机制纯粹带来的迁移能力;如果没有明显差异甚至更低,则说明 contrast 机制在通用数据上就能学到跨任务的 math 能力提升,不依赖训练数据的题材匹配。

   - **第十六轮(2026-07-15,导师建议:新增"question vs irrelevant text"信号,做 case 级互补性分析,为信号合并铺路)**:目前 contrast 机制的 ctrl 是黑图(hi=真图+真问题,ctrl=黑图+真问题),`log p_hi − log p_ctrl` 度量的是**图像依赖**。导师建议**新增一路平行实验**(不是替换 black):ctrl 用**保留真实图像、把问题换成无关文本 "What is the answer?"**(hi=真图+真问题,ctrl=真图+无关问题),同一个差值的语义变成**问题依赖**。

     **目的不是比总分高低,而是 case 级互补性分析**:两个版本(black 版已有,qtext 版新训)在同样的 benchmark 上逐题对比——哪些 case 被 black(图像依赖信号)改进、哪些被 qtext(问题依赖信号)改进、重叠多少。如果两个信号改进的 case 集合明显互补(各自改一批不同的题),**后续就可以想办法把两路信号合并**(比如两个 logprob-diff 的加权组合、或双 ctrl 各算一份 target 再融合),有希望拿到叠加收益;如果高度重叠,合并就没有意义,两个信号本质上是同一个"信息缺失"信号的不同表面。

     **实现零新代码**:现成的 `ra_ctrl_mode=qvis`(保图换文本)+ 把 `ra_generic_prompt` 从默认的 "Describe this image in detail." 换成 "What is the answer?"。与第十一轮 contrast-标准(black)完全同配置(2B、α=1.0 无门控、本仓库数据、62步),唯一变量是 ctrl 构造。**已排队(2026-07-15,trial 301783374,`experiment_script/run_contrast_standard_qtext.sh`,`EXPERIMENT_NAME=Vision-OPD-contrast-standard-qtext-Qwen3-VL-2B-Instruct`)**,跑完自动 merge+9-benchmark+ZoomBench canonical。

     **评测完成后的分析步骤(比训练本身更重要)**:两个版本的评测产物都是逐题记录(VLMEvalKit 的 xlsx/csv 逐题输出+ZoomBench 的 judge jsonl),按 benchmark 逐题拆成四象限:both-correct / black-only-correct / qtext-only-correct / both-wrong,重点看 black-only 和 qtext-only 两格的大小和题目特征(black-only 预期偏细粒度视觉题如 VStar/HRBench,qtext-only 预期偏"问题理解/指令遵循"型题)。基线参照:contrast-标准(black) 7-bench 70.65;历史 qvis 67.20(旧权重机制+默认描述文本,双变量不同,只能参考)。

### Phase 2b:GRPO compute-matched 对比(当前最紧急)

**目的**:回答"给定和 noimg 完全相同的算力(62 步/1 epoch/`train_answer.parquet`),GRPO 单独训练能不能打过/打平 RA-VAD"——唯一能判断"蒸馏机制本身有没有价值"的数字,现有 585 步/3 epoch 对比因算力差 9.4 倍不能用来下结论。

**同步做,几乎零额外成本**:训练过程中每隔 10-20 步存 checkpoint 评测,画"步数 vs 效果"曲线。即使最终 62 步下 GRPO 反超,若 RA-VAD 能用远少于 GRPO 收敛所需步数达到接近效果,这本身是站得住的贡献(训练效率而非渐近上限)。

**防御性论点(不依赖实验结果,现在就写进 outline)**:GRPO 需要可验证 reward,RA-VAD 全程 `reward_model.enable=False`。若下游任务没有干净可验证 reward,GRPO 无法直接套用,RA-VAD 可以。

**进度更新(2026-07-08,最终状态)**:效率曲线目标 8 个 checkpoint(60/100/160/200/260/300/360/400/585,4B,`grpo_vanilla_qwen3vl4b_3ep`)——**60/260/400/585 四点已出完整 7-benchmark 结果,100/160/200/300/360 五个中间点经确认不再补跑(决策见下)**。

| Benchmark | 4B base(`Qwen3-VL-4B-base-clean`) | step60 | step260 | step400 | step585(终点) |
|---|---:|---:|---:|---:|---:|
| BLINK | 64.70 | 65.65 | 66.81 | 66.07 | 63.28 |
| MMStar | 65.47 | 70.20 | 71.13 | 71.20 | 69.13 |
| MMBench_DEV_EN | 73.28 | 82.56 | 85.22 | 82.82 | 85.48 |
| VStarBench | 80.63 | 88.48 | 89.53 | 87.43 | 89.01 |
| MathVista_MINI | 73.30 | 75.70 | 75.70 | 76.70 | 75.70 |
| HRBench4K | 79.88 | 81.00 | 81.25 | 81.25 | 80.88 |
| HRBench8K | 73.25 | 78.13 | 78.00 | 78.75 | 79.50 |
| **7-benchmark 平均** | **72.93** | **77.39** | **78.23** | **77.75** | **77.57** |

**观察**:GRPO 在训练早期(仅占 585 步的 ~10%)就已经比 4B base 提升 4.46 分(77.39 vs 72.93),说明这个纯 GRPO baseline 收敛很快;step260 是四点里的峰值(78.23),400/585 反而略降,**77-78 区间内呈饱和、非单调**的形状,不是持续爬升。

**5 个中间点(100/160/200/300/360)决策(2026-07-08)**:已确认的四点(60/260/400/585)已经足够勾勒出"早期快速提升 + 之后饱和/轻微震荡"这个定性形状,补齐中间点对当前结论(GRPO compute-matched 对比、以及和 noimg 62 步/68.20 平均分的差距量级)边际信息量有限,**决定不再补跑**,不作为阻塞项。

**与 noimg 的比较**(仍不是 compute-matched,仅供参照):noimg(62步/1epoch)7-benchmark 平均 68.20,明显低于 GRPO 在同等 62 步区间的 77.39(step60)。这个差距的解释需要谨慎:GRPO 用的是 `data/train.parquet`(纯 MCQ,有 verifiable reward),noimg 用的是 `data/train_answer.parquet`,两者数据分布不同,而且 60 步时 GRPO 的 batch/rollout 配置(`train_batch_size=32`)也和 noimg(`train_batch_size=96`)不一样,实际"看过的样本数"不对等——**这仍然不是严格的 compute-matched 对比**,只能定性地说"在这个数据分布上,纯 GRPO 早期收敛很快",不能直接得出"GRPO 优于 RA-VAD 蒸馏机制"的结论。

### Phase 2c:GRPO + RA-VAD 结合(中优先级,视 2b 结果决定)

现有循环已是 `algorithm.adv_estimator=grpo` + `loss_mode=vopd` 的"二选一",改成"两者都算"不需要重搭框架。

**方案 A(先做,风险低)**:`L_total = L_GRPO + β · L_RA-VAD`,GRPO 动力学不变,RA-VAD 作为正则项叠加,β 扫 0.1/0.5/1.0。

**方案 B(二期,novelty 风险需认真处理)**:用 `ra_weight` 直接调制/门控 GRPO policy-gradient。这条路(尤其"gap 高用 KL、低用 GRPO"这种硬阈值门控)**和 AOPD(arXiv 2605.06387)高度相似**——AOPD 按 teacher-student advantage 分区,advantage 为正保留 PG,非正切换成 forward KL,且这是一个拥挤子领域(ReLIFT/SASR/BRIDGE/SRFT/Prefix-RFT/LUFFY/CHORD)。**差异化点**:AOPD 门控依据是"模型哪里能力不足"(competence gap),纯文本 reasoning,你的 `ra_weight` 门控依据是"任务哪里需要看图"(modality dependence)——正交轴,不是同一个东西的重新包装,但方案 B 必须把这点讲清楚。可借鉴 AOPD 的两条工程结论(forward KL 优于 reverse KL、用 top-K 而非全词表),不算 novelty 冲突。

### Phase 2 优先级汇总

| 子项 | 成本 | 阻塞什么 | 优先级 |
|---|---:|---|---|
| 2-核心:纯 EMA 对照组 X | 中(1组62步训练) | 组件分解:EMA vs 视觉信号 | **最高,与 2b 并列成稿前必做** |
| 2a shuffle | 低(代码就绪+短训练) | 组件分解:视觉相关性 vs 泛加权 | **最高,与 2b 并列成稿前必做** |
| 2b GRPO compute-matched | 中(62步训练,已在做) | 蒸馏机制本身有无价值 | **最高,当前最紧急** |
| ZoomBench 补 v2-fixed 重判(black/degrade/qvis/virl39k/sr1) | 低(仅重跑judge,零训练) | 8-benchmark composite 表能否成立、SR1 short-board 论证是否站得住 | **最高,零训练成本,应立刻做** |
| 2c 方案A | 低-中 | 无 | 中,视2b结果决定 |
| 2c 方案B | 中-高 | 无 | 低,二期方向 |

---

## Phase 3: ViRL39K / Vision-SR1 外部数据集迁移 ✅

**动机**:验证"noimg 这套 RA-VAD self-distillation 方法是否依赖特定训练数据分布"。用 TIGER-Lab/ViRL39K(38327 条,几何/图表 MCQ 为主)、LMMs-Lab-Turtle/Vision-SR1-47K(47628 条,通用 VQA)转换成同样的 noimg schema 重新训练,同一套 VLMEvalKit 7-benchmark 评测对比;ZoomBench 部分见下方警示。

### 3.1 ckpt1k+ 结果(单点,训练至收敛):noimg-virl39k-2B(step1197)/ noimg-vision-sr1-2B(step1480)

> ✅ **口径已更新(2026-07-07 晚间)**:virl39k/sr1 已用 Phase 4 三重修复后(v3-fixed)的 native pipeline 重新推理+重判,下表 ZoomBench 列已替换成正式口径数字。这两个模型走的是标准 native eval(非 thinking, temp0, max_tokens 8192),表里其余行(base/VisionOPD 系列)仍是旧的 Azure-fix-但未过 Phase4 三重修复的历史数字,只有 virl39k/sr1 两行是当前口径,不能跨行直接比较,只能看 virl39k vs sr1 这一组内部的相对高低。

| Model | ZoomBench Acc |
|---|---:|
| Qwen3-VL-2B-Instruct(base) | 40.36%(旧口径,历史参照) |
| Qwen3-VL-4B-Instruct(base) | 41.18%(旧口径,历史参照) |
| VisionOPD-baseline-2B(step65) | 36.45%(旧口径,历史参照) |
| VisionOPD-noimg-2B(step62) | 37.63%(旧口径,历史参照) |
| VisionOPD-black-2B(step65) | 36.45%(旧口径,历史参照) |
| VisionOPD-degrade-2B(step62) | 37.40%(旧口径,历史参照) |
| VisionOPD-qvis-2B(step62) | 37.51%(旧口径,历史参照) |
| VisionOPD-full-2B(step65) | 40.12%(旧口径,历史参照) |
| **noimg-virl39k-2B(step1197)** | **38.93%**(✅ v3-fixed 正式口径,原38.58%,几乎无变化) |
| **noimg-vision-sr1-2B(step1480)** | **39.64%**(✅ v3-fixed 正式口径,原40.24%,几乎无变化) |

**观察**:和 3.1 之前"sr1 > virl39k"的排序在 v3-fixed 后依然成立(39.64% vs 38.93%),且两个数字变化都很小(±0.65pp 以内)——说明 virl39k/sr1 这两个 checkpoint 的答案格式本来就比较简短、没怎么踩上三个 judge bug(不像 noimg-4B 那样因为回答长度暴涨 20 倍而被 Bug 3 严重误伤)。

**VLMEvalKit 7-benchmark**(temp0/4096,和两个新模型同一套 judge/pipeline,可直接横向比较,不受 ZoomBench 口径切换影响):

| model | BLINK | MMStar | MMBench_DEV_EN | VStarBench | MathVista_MINI | HRBench4K | HRBench8K |
|---|---:|---:|---:|---:|---:|---:|---:|
| noimg(本仓库数据,step62) | 55.50 | 61.20 | 76.55 | 73.82 | 63.20 | 75.75 | 71.38 |
| qvis(step62) | 53.34 | 61.20 | 75.43 | 71.20 | 62.70 | 75.25 | 71.25 |
| black(step65) | 54.71 | 60.33 | 76.46 | 73.82 | 63.60 | 75.25 | 70.63 |
| **noimg-virl39k(step1197)** | 54.18 | 59.60 | 73.88 | 68.59 | 62.10 | 72.13 | 68.00 |
| **noimg-vision-sr1(step1480)** | 51.92 | 59.53 | 76.03 | 69.11 | 61.10 | 71.25 | 69.63 |

两个外部数据集模型在 7-benchmark 上普遍略低于本仓库自有数据训练的 noimg/qvis/black。

### 3.2 v2 训练 step 扫描(virl39k-v2:step50/150/300/500;sr1-v2:step50/150/250)——**目前全篇最强的"短板来自数据不来自机制"证据**

| Checkpoint | BLINK | MMStar | MMBench_DEV_EN | VStarBench | MathVista_MINI | HRBench4K | HRBench8K | **Avg(7-bench)** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline(step65) | 55.44 | 58.87 | 69.24 | 72.25 | 61.80 | 73.25 | 69.63 | 65.78 |
| noimg VisionOPD(本仓库数据,step62) | 55.50 | 61.20 | 76.55 | 73.82 | 63.20 | 75.75 | 71.38 | **68.20** |
| noimg-virl39k-2B-v2 step50 | 53.55 | 60.40 | 75.26 | 73.30 | 62.60 | 75.25 | 72.00 | 67.48 |
| noimg-virl39k-2B-v2 step150 | 54.23 | 61.00 | 76.63 | **76.44** | 62.80 | 74.88 | 71.00 | 68.14 |
| noimg-virl39k-2B-v2 step300 | 54.18 | 59.67 | 75.52 | 71.20 | 63.10 | 73.13 | 69.13 | 66.56 |
| noimg-virl39k-2B-v2 step500 | 54.13 | 61.67 | 75.26 | **75.92** | 63.70 | 74.75 | 70.88 | 68.04 |
| noimg-vision-sr1-2B-v2 step50 | 53.92 | 61.33 | 75.95 | 71.20 | 64.10 | 74.38 | 70.75 | 67.38 |
| noimg-vision-sr1-2B-v2 step150 | 53.60 | 61.73 | 77.06 | 71.20 | 62.50 | 76.13 | 71.88 | 67.73 |
| noimg-vision-sr1-2B-v2 step250 | 55.39 | 60.93 | 75.69 | 73.82 | 62.90 | 76.00 | 71.38 | 68.02 |
| noimg-virl39k-2B-**filtered** step464(1 epoch,offline/server 双跑互相校验,差≤1.6pp/单项、0.03pp/平均) | 54.45~55.55 | 60.67~61.33 | 74.31~74.91 | 73.30~74.87 | 61.10~61.50 | 76.38~76.62 | 69.50~70.25 | **67.47~67.50** |

**virl39k-filtered(2026-07-09~10)补充说明**:用离线多样本 rollout 过滤后的 virl39k 数据、464 步(比 v2 扫描的 step500 更长)训练,7-bench 综合分(67.47-67.50)和 v2 系列的 step150/500(68.14/68.04)、标准 noimg(68.20)基本还是同一档,没有系统性突破,但 **ZoomBench(native v3-fixed pipeline)明显更好:41.54%**,是目前测过的所有 noimg-virl39k 变体里最高的(v2 step150/step500/step1197 分别是 37.87%/39.53%/38.93%),已经很接近 base-2B(42.49%)。**结合上面的 VStarBench 证据一起看**:数据过滤 + 更长训练,在 search/zoom 类任务上确实有实打实的提升空间,和"综合感知类任务已经打平、增益空间有限"这两件事并不矛盾——两者分别对应"短板来自数据分布"和"token 加权本身贡献有限"这两条独立的结论。

**关键新证据**:**virl39k-v2 step150(76.44)和 step500(75.92)的 VStarBench 都超过了 visionopd(full) 的 75.39**——这是全部结果里第一次有 noimg 机制的变体在 search 类 benchmark 上**正面超过** bbox-crop 方法,而且是统一 VLMEvalKit pipeline、不受任何 judge bug 影响的干净证据。**这应作为"ZoomBench/VStar 短板来自训练数据分布、不是机制本身局限"这条论证的主证据**,比 3.1 里 SR1 的 ZoomBench 数字(待重判)更硬,应在 Analysis 章节优先引用。

**其他观察**:
- 本仓库原始 noimg(68.20)在这个对比里仍是全表最高,但差距很小,virl39k-v2 step150(68.14)/step500(68.04)/sr1-v2 step250(68.02)都在 0.2pp 以内追平,结合 0.8b 测到的 composite 种子噪声(±0.38),三者统计上打平——**结论是方法在三个不同数据分布上都稳定有效,不是过拟合到本仓库数据,但不能 claim"我们的数据更好"**。
- virl39k-v2 没有单调收敛趋势(step300 是明显低点),提示训练过程本身噪声较大;sr1-v2 更接近单调爬升,可能还有继续提升空间。
- 所有外部数据集 checkpoint 都明显优于 baseline(65.78),支持"noimg self-distillation 方法本身"换数据后依然有效。

### 3.3 4B 规模验证:noimg-4B(Vision-OPD-noimg-Qwen3-VL-4B-Instruct,step62)

> ⚠️ 下表沿用官方 pipeline(旧口径),需要时应补 v2-fixed 版本;4B base/noimg-4B 的 v2-fixed 数字见 Phase 4 表格。

| Model | ZoomBench(官方 pipeline,旧口径) |
|---|---:|
| Qwen3-VL-4B base | 41.07% |
| Qwen3-VL-2B base | 37.40% |
| noimg-2B(step62) | 37.16% |
| **noimg-4B(step62)** | **36.57%** |

noimg-4B 比 noimg-2B 还略低,也明显低于 4B base 自己,延续了 2B 上的模式,换成 4B 规模同样成立。ZoomBench 的 v3-fixed native pipeline 数字见 Phase 4(noimg-4B = 39.76%)。

**✅ VLMEvalKit 7-benchmark 结果(2026-07-08,已完成,更新此前"评测跑中"的过时状态)**:

| Benchmark | noimg-4B(step62) |
|---|---:|
| BLINK | 66.86 |
| MMStar | 69.47 |
| MMBench_DEV_EN | 83.08 |
| VStarBench | 82.72 |
| MathVista_MINI | 77.00 |
| HRBench4K | 79.50 |
| HRBench8K | 74.75 |
| **7-benchmark 平均** | **76.20** |

baseline-4B、VisionOPD-full-4B 的同套干净 7-benchmark 结果正在补跑(2026-07-08,GPU5/GPU1,`Qwen3-VL-4B-base-clean`/`VisionOPD-full-4B-step65-clean`,此前共享输出目录里同名旧结果经核实是配置污染的 2B 数据,已弃用),出来后可以拼出完整的 4B 三方对比表,判断 scale 故事是否成立。

---

## Phase 4: native eval judge 三 bug 排查与修复(现为 ZoomBench 正式口径)✅

**起因**:VisionOPD-4B 的 native eval(`Vision-OPD/eval/` 的 infer→judge_qwenlm→cal_acc 流程)在不同时间点测出 49.41%、34.20%、36.22% 三个互相矛盾的数字,同一批模型预测换个 judge 就能差 10-15pp。排查后确认不是模型/ckpt 问题(逐字比对预测,78-96% 一致),而是 `judge_qwenlm.py` 里的**三个独立 bug**(前两个在初次排查时发现,第三个是后来做样本级对比分析时意外挖出来的)。

### Bug 1:`MCQ_BENCHMARKS` 漏掉了 `zoombench`

MCQ 快速通道(`first_letter_match` 能判定就不调用 LLM)只对 `MCQ_BENCHMARKS` 列表里的 benchmark 生效,`zoombench` 一直不在列表里——845 题里 620 道 MCQ 题(其中 357 道字母+内容完全匹配、本该被规则直判)全被送去不稳定的 LLM judge。人工核验 140 条:所有 38 条"规则 vs judge"分歧案例,规则判定 38/38 全部正确;`gpt-5.4-mini` 在这些分歧里 33/33 都是假阴性(尤其带 markdown/`✅`/加粗格式的答案容易漏判)。

**修复**:`MCQ_BENCHMARKS` 加入 `"zoombench"`。

### Bug 2:`extract_first_option()` 抓"第一个"而不是"最后一个"字母

修复 bug 1 后新出现的问题:走规则通道的题目提取逻辑抓文本里**第一个**大写字母。当模型输出是"逐项分析 A/B/C/D 再给结论"的长 CoT,或 `"Answer: X**"` 格式("Answer" 本身以大写 A 开头),规则会抓错字母。当 GT 恰好是 A 时,抓错字母又"巧合"等于 GT,造成假阳性。人工审计:noimg-4B 的 225 条规则判定里 22 条(9.8%)确认假阳性,VisionOPD-4B 的 358 条里只有 1 条。

**修复**:改成优先匹配"correct answer is X"/"answer: X"等结论性标记(取最后一次出现),找不到则退化为取文本里最后一个枚举出现的选项字母,而不是第一个。

### Bug 3(2026-07-07 晚间新发现):开放式数字题的长 CoT 提取失败,系统性误伤"话多"的模型

在做 base/VisionOPD-4B/noimg-4B 三方样本级对比(逐条排查退步/进步样本)时发现:noimg-4B 有大量数字题("Please answer using Arabic numerals")的模型回答**其实和 GT 完全一致**,却被判"No"。根因:`extract_answer()` 只在文本含 `<answer>` 标签或 `"Answer:"` 关键词时才做截断,如果模型的数字结论是"……therefore the answer is 3.\n\n3"这种没有固定前缀的自由格式,`extracted_answer` 会变成整段长 CoT 原文,喂给 `mathruler.grade_answer(gt, extracted_answer)` 几乎必然匹配失败,于是被送去 LLM judge——而 LLM judge(`gpt-5.4-mini`)在长文本里找数字同样不可靠。

**这个 bug 是不对称的**:审计发现 noimg-4B 的 194 条被 LLM 判"错"的数字题里,**37 条(19%)规则重新提取后数字其实和 GT 一致**;而 base、VisionOPD-4B 的同类假阴性都是 **0 条**。原因是 noimg-4B 的回答平均长度是 base 的 ~20 倍(1996 字符 vs 97 字符)——**答案越啰嗦,被这个 bug 误伤的概率越高**,这直接放大了 noimg 看起来"更差"的假象。

**修复**:新增 `extract_final_number()` + `numeric_match()`,和 Bug 2 同一思路——优先匹配"answer is X"式结论标记(取最后一次出现),找不到则退化为取文本里最后一个出现的数字,而不是丢给 mathruler 处理整段原文。

### 三重修复后的 6 模型 ZoomBench 对照(845 题,native pipeline,**这是当前锁定的正式口径**)

| 模型 | 修复前(bug污染) | v1(仅Bug1) | v2(Bug1+2) | **v3(Bug1+2+3,最终,锁定口径)** | 旧官方 pipeline(历史参照) | Gap(v3−旧官方) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-4B-base | — | 44.14% | 44.14% | **44.14%** | 41.07% | +3.07pp |
| Qwen3-VL-2B-base | — | 42.84% | 42.84% | **42.49%** | 37.40% | +5.09pp |
| VisionOPD-4B | 34.20%/49.41%(矛盾) | 53.14% | 52.90% | **52.90%** | 48.76% | +4.14pp |
| VisionOPD-2B(full) | 19.53%(旧buggy) | 34.08% | 35.50% | **37.51%** | 40.00% | −2.49pp |
| noimg-4B | 18.93%(mini,bug) | 32.31% | 35.38% | **39.76%** | 36.57% | +3.19pp |
| noimg-2B | 16.80%/18.93%(旧buggy) | 32.19% | 31.72% | **36.45%** | 37.16% | −0.71pp |

**观察**:
- Bug 3 修复带来的变化集中在**答案更长的模型**:noimg-4B(35.38%→39.76%,+4.38pp)、noimg-2B(31.72%→36.45%,+4.73pp)、VisionOPD-2B(35.50%→37.51%,+2.01pp)明显上升;而 base 系列(答案本来就短)几乎不变(4B-base 持平,2B-base -0.35pp 波动)。
- 修复前 native pipeline 的数字混乱、互相矛盾,看似"全面比旧官方 pipeline 低 15-20pp",这本身是 judge bug 的假象。三重修复后,6 个模型和旧官方 pipeline 的差距全部收窄到 **±5.1pp 以内**,且方向不再一致偏低(noimg-4B、4B-base 甚至反超旧官方 pipeline)。
- **VisionOPD-4B 从 34.20%/49.41% 的矛盾数字收敛到稳定的 52.90%**;noimg-4B 从 18.93%(第一次buggy) 一路收敛到 **39.76%**,是三次修复里改动幅度最大的模型,也印证了它此前"特别差"的结论主要是 judge 假象,不是模型真的这么差。

### noimg-4B vs base vs VisionOPD-4B 样本级对比:"noimg 为什么比 base 差"(2026-07-07)

用 v3-fixed 判分文件逐样本对比(845 题全量):

| | vs base 退步 | vs base 进步 | 净变化 |
|---|---:|---:|---:|
| VisionOPD-4B | 40 | 114 | **+74** |
| noimg-4B | 95 | 58 | **−37** |

修复 Bug 3 前(v2 口径)净变化是 -74,**修复后腰斩到 -37**——说明之前"noimg 特别差"的结论里有近一半是判分假象,真实退步是这个数字的一半。

对剩余 95 条真实退步样本按问题类型分类(vs 845 题全体分布做归一化对比):
- **时间读取(clock)**:放大 2.45x(全体占比 3.4%,退步样本占 8.4%)——**唯一显著、修复后依然存在的短板**。人工读了全部 8 条案例:base 直接给出自信答案(疑似模式匹配、不推理),noimg 认真做 CoT"数指针位置"但精度系统性变差(如把 5:02 读成 5:10、11:24 读成 11:15),还出现过一次显著的重复循环退化("the hour hand is pointing to the 6? No —" 重复 9+ 次)。
- **颜色(color)**:放大 1.10x,较轻,抽查确认是真实的细粒度颜色混淆(white vs off-white、gray vs blue),不是判分问题。
- **计数(counting)**:放大从修复前的 1.63x 降到 **1.02x**(基本回归正常),证实之前的"计数能力下降"结论主要是 Bug 3 造成的假象。
- **文字/OCR/品牌**:放大仅 0.16x,noimg 在这类任务上完全没有退步。

**结论**:noimg 在 ZoomBench 上比 base 差,真实原因不是感知能力全面退化,而是**局限于需要高精度读数(钟表指针角度)的任务上系统性变粗糙**,同时回答变长(2000 字符 vs 97 字符)放大了判分管线的误判——这部分已经修复,真实差距只有表面数字的一半左右。建议论文里用这个更精确、可证伪的样本级解释,而不是笼统的"分布外任务灾难性遗忘"。

**代码改动**:三处修复已落地在 `Vision-OPD/eval/judge_qwenlm.py`(`MCQ_BENCHMARKS` 列表 + `extract_first_option()` + 新增 `extract_final_number()`/`numeric_match()`),可复现快照见 `eval/judge_qwenlm_zoombench_fixed_20260707.py`,对未来所有 zoombench native eval 自动生效。

---

## 如果之后想更进一步(暂不建议现在做)

除了"只换加权机制"的收紧版本,理论上还有更彻底的复现方式:新增严格复刻 VA-OPD 的 control、换成外部大模型 teacher、换到 Geometry3K/ViRL39K 数据和论文的 8 个 benchmark 逐字复现。这一档价值在于"验证实现和原论文数字对得上",但对论文核心 claim 边际贡献小、成本高得多。既然明确不换 teacher,这一档直接排除,只有审稿人明确要求时才回头考虑。

ZoomBench 上正面拿下 VisionOPD:Phase 3.2 的 virl39k-v2 VStarBench 反超已经是很强的证据("换数据就能追近/反超"),不需要再单独设计实验去"正面对抗",这条已经完成得差不多。

Detail-augment(Core∪normalized(degrade))训练组:Phase -0.5 只做过离线诊断,没有真正训练过,写成 future work,不要现在训练。

---

## 各阶段的比较

| | -0.5 | -0.3 | 0 | 0.8a | 0.8b | 1 | 1.5 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|---|---|---|---|
| 训练成本 | 零 | 零 | 零 | 零 | 中(已完成) | 中(已完成) | 零 | 中(进行中) | 中(已完成) | 零(已完成) |
| 状态 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🔄 | ✅ | ✅ |

**结论**:三个前置诊断(-0.5、-0.3、0)+ 0.8a + 0.8b 已全部出结果,支持"noimg 是最佳主 control、暂不需要 signed variant",7-benchmark composite 层面对训练种子稳健;Phase 1 的 5 组训练评测全部完成,用 0.8b 测出的噪声本底重新校验 7-benchmark 部分后:**`vaopd-grouped-rollout-forward` 是唯一一个多数 benchmark 都显著超出噪声本底的变体,是真实、全面的退步**;其余变体和 noimg 基准差异大多在噪声范围内或有限的真实 trade-off。Phase 3 证明方法在外部数据(ViRL39K/Vision-SR1)上依然有效,且 virl39k-v2 提供了"换数据即可在 VStarBench 反超 visionopd"的强证据,支持"search 短板来自数据分布、不是机制"。Phase 4 排查并修复了 native eval 的两个 judge bug,现在是 ZoomBench 的锁定口径,但也带来一个需要更新的认知:**noimg 和 visionopd(full)在 v2-fixed 口径下相对 base 都是退步,不是"一个持平一个提升"**。

**当前唯二真正阻塞论文成稿的事项**:
1. **训练类**:Phase 2b(GRPO compute-matched)+ Phase 2-核心(三组件贡献分解,纯EMA对照组X + shuffle),三个 62 步训练,回答"和外部方法比行不行"以及"内部哪个组件在起作用"。
2. **零训练成本,但同样紧急**:用锁定的 v2-fixed pipeline 重判 black/degrade/qvis 和 Phase 3 的 virl39k/sr1 的 ZoomBench,补全 8-benchmark composite 表,并确认 Phase 3.1 的 SR1 排序在新 judge 下是否依然成立。

### 下一步

1. **零训练成本,立刻做**:用锁定的 v2-fixed pipeline 重判 black/degrade/qvis(补全 Vision-OPD 2B 主表)、virl39k/sr1(确认 Phase 3.1 排序是否变化)的 ZoomBench;重算所有 8-benchmark composite 表。
2. **成稿前必做,训练类,三组并列**:
   - Phase 2-核心的三组件贡献分解(纯 EMA 对照组 X + shuffle 组,代码均已就绪)。
   - Phase 2b 的 62 步 compute-matched GRPO 对比(已在做),以及训练效率曲线。
   - noimg-4B 的 VLMEvalKit 7-benchmark(评测跑中),决定 scale 故事怎么写。
3. **视 2b 结果决定**:Phase 2c 方案 A(加法组合),以及是否投入方案 B(需先写清楚和 AOPD 的差异化论点)。
4. **锁定最终 loss 设计**:主 control 用 `noimg`,权重机制维持现状连续加权;散度上 reverse KL 和现状统计上无法区分,可作等价备选;JSD 不建议默认;硬分组+rollout reweight 全机制版本明确不采用。
5. **论文骨架**:Intro → Related Work(含 VA-OPD/ZwZ/AOPD 差异化定位)→ Method → 为什么是这四种 control(Phase -0.5)→ 主结果(7-benchmark 稳定,ZoomBench 待补齐 v2-fixed 后再定案)→ Ablation(Stage 1 权重/散度 + 三组件贡献分解)→ Analysis(通用性 vs 窄向过拟合;同数据不同用法;virl39k-v2 VStar 反超作为主证据,SR1 作辅助待确认)→ Limitations(4B 7-benchmark 待补、Detail-augment 待验证)。