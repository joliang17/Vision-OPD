# 论文写作素材独立整理（Codex）

> 生成日期：2026-07-15  
> 范围：仅基于 `docs/reports/ra_vad_contrast_report.html`、`docs/compare_vaopd_0701.md`、`docs/experiment_settings_contrast_standard_virl39k_90step.md`、`docs/queued_experiments_20260713.md`、`docs/eval_integrity_registry.md` 独立整理。  
> 注意：本文是写 paper 用素材库，不是最终论文草稿；所有数字引用前仍应按 `eval_integrity_registry.md` 和最新评测输出复核一次。

## 一句话主线

RA-VAD/contrast target 路线可以写成一种**无外部 teacher、无标注 reward、无 RL reward 的视觉条件自蒸馏方法**：用 EMA teacher 在真实图像条件和视觉/文本控制条件下对同一条 student rollout 做 teacher-forcing 打分，利用两种条件下的逐 token log-prob gap 同时构造 token 权重和对比锐化蒸馏 target。最强的 2B 版本不是早期纯 `noimg` 权重，而是后续的 **contrast-标准** 配置：在本仓库数据上 7-bench 平均 70.65，较 2B baseline 66.05 提升 +4.60pp，较 noimg 68.20 提升 +2.45pp；在 `virl39k-filtered` 90 步短训上 7-bench 70.68，ZoomBench canonical 41.89。

推荐论文叙事：

1. 先介绍早期 RA-VAD 的观察：单纯把 hi/control gap 当 token 权重有收益，但主要收益来自 EMA 自蒸馏，视觉相关性权重只是增量。
2. 再引出 contrast target：不只重加权 teacher 分布，而是把 `p_hi / p_ctrl` 的对比信息蒸馏进 target 本身。
3. 主结果强调通用感知 benchmark 的提升；ZoomBench/search 类任务上 bbox-crop VisionOPD 仍有优势，尤其 4B 上明显。
4. limitation 诚实写：训练长度有安全区，长训会自增强失控；严格多步数学推理有副作用；评测 pipeline/prompt 风格对 HRBench/ZoomBench 很敏感。

## 方法描述素材

### 基础 RA-VAD

可写成：

给定 student 采样得到的 response token 序列，EMA teacher 不重新生成答案，而是在两种输入条件下对同一条 rollout 做 teacher forcing：

- hi 条件：真实图像 + 原问题。
- control 条件：按实验不同，可为 noimg、同尺寸黑图、降质图、保图换文本等。

对第 `t` 个 response token，计算 chosen-token log-prob gap：

```text
a_t = log p_T(y_t | y_<t, hi) - log p_T(y_t | y_<t, ctrl)
```

早期 RA-VAD 把正的 gap 视为“该 token 依赖视觉条件”的证据，经 ReLU、逐行 95 分位裁剪、正值均值归一化和样本级门控后得到 detached token 权重 `w_t`，再做加权 forward-KL 蒸馏。

关键定位：这不是 RL，也不是用更强 teacher；信号来自模型自身在输入扰动下的敏感度。可以与 Vision-OPD 的 bbox-crop teacher、VA-OPD 的外部大 teacher、GRPO 的可验证 reward 区分开。

### Contrast-sharpened target

后续最强版本把 gap 从“只做权重”升级为“改写蒸馏 target”。在全词表上构造：

```text
tilted(w) = log p_hi(w) + alpha * (log p_hi(w) - log p_ctrl(w))
target = softmax(tilted) over plausibility support
```

等价于几何倾斜：

```text
target(w) ∝ p_hi(w)^(1+alpha) / p_ctrl(w)^alpha
```

实现细节应写清：

- 使用温度 `T=2` 软化 teacher/student 分布，loss 乘 `T^2`。
- `beta=0.1` plausibility mask：只保留 `p_hi >= 0.1 * max(p_hi)` 的候选 token，避免把 hi 条件下本来不可信的 token 放大。
- `<|endoftext|>` 和 `<|im_end|>` 是“tilt 豁免”，不是从 target 中删除；它们保留原 `lp_hi` 打分，仍受 plausibility mask 约束。
- 标准配置：`alpha=1.0`，position gate 关闭，即所有位置都做 target 倾斜。
- 保守配置：`alpha=0.5`，position gate 开启，只在 `w_t > 0` 的位置倾斜。
- 外层 token 权重仍保留。uniform-weight 消融显示去掉外层权重后 7-bench 从 70.65 降到 68.97，下降 1.68pp。

### 训练配置可复现口径

`contrast-标准 × virl39k-filtered 90step` 的可复现描述：

- Base：Qwen3-VL-2B-Instruct，HF snapshot `89644892...`。
- Teacher：student 权重的 EMA，`teacher_update_rate=0.05`，即 `theta_T <- 0.95 theta_T + 0.05 theta_S`。
- 数据：`virl39k_train_noimg_filtered_1img.parquet`，单图子集 14,002 条；原 filtered 版 14,861 条中 859 条多图样本被剔除，因为全词表蒸馏 backward 会确定性 OOM。
- GPU：4 x B200 183GB，FSDP，param/optimizer offload。
- 训练：90 optimizer steps，batch size 32 prompts/step，rollout.n=8，LR 2e-6，10-step warmup，constant schedule。
- 长度：max prompt 6144，max response 1024，rollout max model len 7168。
- Loss：`loss_mode=vopd`，`reward_model.enable=False`，`use_kl_loss=False`；actor 总 loss 即蒸馏 loss。
- 全词表蒸馏：`full_logit_distillation=True`，`distillation_topk=null`。RA-VAD 路线不能用 top-k 省算力。
- 评测主口径：VLMEvalKit server，temperature 0，top_p 0.8，top_k 20，presence_penalty 1.5，max_new_tokens 4096，judge 为 Azure `gpt-5.4-mini-2026-03-17`；ZoomBench 用 canonical/native GPT judge 口径。

复现风险：

- `ANSWER_VAL_TRAIN_FILE` 在特定 mtime/文件缺失条件下可能被 split 初始化逻辑静默重写；复现命令最好显式禁用自动 split 或先修脚本。
- `resume_mode=auto` 是默认值；同名 `EXPERIMENT_NAME` 重跑可能自动续训，不是从头开始。
- black ctrl 应描述为“保留图像占位符、尺寸和视觉 token 数，只把内容替换为纯黑 RGB”，不要写成 no-image。

## 主结果数字

### 2B 本仓库数据主表

可引用的 2B 结果（以报告 HTML 的主表为准）：

| 模型 | 数据/设置 | 7-bench | POPE | Hallusion aAcc/fAcc/qAcc | ZoomBench |
|---|---|---:|---:|---:|---:|
| baseline | 原始未训练 | 66.05 | 88.76 | 68.35 / 41.62 / 44.84 | 42.49 |
| VisionOPD | bbox-crop teacher, step65 | 66.20 | 未测 | 未测 | 37.51 |
| noimg | 纯 hi target, step62 | 68.20 | 89.16 | 66.98 / 43.06 / 43.96 | 36.45 |
| GRPO | `train.parquet`, step585/3ep | 70.19 | 87.64 | 66.14 / 41.04 / 43.30 | 47.81 |
| contrast-保守 | 本仓库, alpha=0.5, gated | 69.07 | 88.41 | 68.56 / 44.22 / 45.93 | 40.59 |
| contrast-标准 | 本仓库, alpha=1.0, ungated | 70.65 | 89.13 | 69.19 / 46.53 / 47.47 | 43.67 |
| contrast-标准-uniform-weight | 去外层 `w_t` | 68.97 | 88.97 | 69.40 / 43.35 / 45.05 | 补跑中 |

直接可写结论：

- contrast-标准在 2B 本仓库数据上是当前最强 RA-VAD/contrast 变体，7-bench 70.65，相对 baseline +4.60pp，相对早期 noimg +2.45pp，相对 VisionOPD 7-bench +4.45pp。
- contrast-标准在 POPE/HallusionBench 也不差：POPE 89.13，Hallusion qAcc 47.47，高于 baseline qAcc 44.84 和 noimg qAcc 43.96。
- GRPO 2B 3ep 的 7-bench 70.19 低于 contrast-标准 70.65，但 ZoomBench 47.81 高于 contrast-标准 43.67。不要简单写“contrast 全面优于 GRPO”；应写“在通用 7-benchmark 上可比或略优，在 search/zoom 任务上 GRPO 更强”。

### 2B 外部数据与 90 步短训

`contrast-标准 × virl39k-filtered 90step`：

- BLINK 58.18
- MMStar 63.87
- MMBench_DEV_EN 78.18
- VStarBench 78.01
- MathVista_MINI 67.00
- HRBench4K 76.38
- HRBench8K 73.13
- 7-bench 70.68
- POPE 88.36
- HallusionBench 68.56 / 44.51 / 45.71
- ZoomBench 41.89

`contrast-保守 × virl39k-filtered 90step`：

- 7-bench 70.21
- POPE 89.10
- HallusionBench 69.40 / 45.38 / 47.03
- ZoomBench 41.18

`contrast-标准 × sr1-filtered 90step`：

- 7-bench 69.65
- POPE 88.78
- HallusionBench 68.98 / 43.93 / 47.03
- ZoomBench 41.66

`contrast-保守 × sr1-filtered 90step`：

- 7-bench 69.12
- POPE 89.24
- HallusionBench 68.77 / 45.66 / 45.71
- ZoomBench 41.66

可写结论：

- 90 步短训验证了训练长度假设：同样 contrast-标准 × virl39k，437 步长训崩到 7-bench 58.31，而 90 步恢复到 70.68，和本仓库数据 62 步的 70.65 基本打平。
- 外部数据并没有带来远超本仓库数据的通用分数上限；更像是证明机制对数据分布有一定迁移性，但训练长度必须受控。
- `virl39k` 对 VStar/Search 更友好：标准×virl39k step90 的 VStarBench 78.01，高于本仓库 contrast-标准 76.96，也高于 2B VisionOPD 75.39。

### 90 步中间点趋势

`标准 × virl39k`：

- step30：7-bench 69.62，ZoomBench 40.95
- step60：7-bench 69.61，ZoomBench 41.66
- step90：7-bench 70.68，ZoomBench 41.89

`保守 × virl39k`：

- step30：7-bench 69.33，ZoomBench 40.36
- step60：7-bench 70.43，ZoomBench 41.89
- step90：7-bench 70.21，ZoomBench 41.18

`标准 × sr1`：

- step30：7-bench 69.69，ZoomBench 42.84
- step60：7-bench 69.62，ZoomBench 42.25
- step90：7-bench 69.65，ZoomBench 41.66

`保守 × sr1`：

- step30：7-bench 69.56，ZoomBench 41.54
- step60：7-bench 69.67，ZoomBench 42.25
- step90：7-bench 69.12，ZoomBench 41.66

注意写法：

- 不要写“90 步最优”。更准确是“90 步以内健康，没有长训崩溃；具体最优点非单调，保守配置在两个数据集上都 step60 见顶”。
- 可以提出假设：保守配置 `alpha=0.5 + gate` 可能更早触及性能天花板；标准配置在 virl39k 上还有上升趋势。

### 4B Qwen3-VL 结果

| 模型 | 7-bench | POPE | Hallusion aAcc/fAcc/qAcc | ZoomBench |
|---|---:|---:|---:|---:|
| baseline 4B | 75.48 | 89.43 | 70.56 / 45.95 / 47.91 | 44.14 |
| VisionOPD 4B | 74.40 | 89.60 | 70.45 / 47.98 / 49.67 | 52.90 |
| GRPO 4B step585 | 77.57 | 87.97 | 71.71 / 50.58 / 49.67 | 53.25 |
| contrast-保守 4B | 77.13 | 88.45 | 73.08 / 52.31 / 53.19 | 43.79 |
| contrast-标准 4B | 76.79 | 88.71 | 73.19 / 52.60 / 52.75 | 44.62 |

可写结论：

- 4B 上 contrast 两个变体在 7-bench 上超过 baseline 和 VisionOPD：保守 77.13、标准 76.79 vs baseline 75.48、VisionOPD 74.40。
- 4B 上 HallusionBench qAcc 明显提升：保守 53.19、标准 52.75，高于 baseline 47.91 和 VisionOPD 49.67。
- 但 ZoomBench 上 contrast 低于 VisionOPD/GRPO：标准 44.62、保守 43.79 vs VisionOPD 52.90、GRPO 53.25。这个是重要 caveat，说明 bbox-crop teacher/search-specific 信号在 zoom/search 类任务上仍有真实优势。

### Qwen3.5-4B 当前材料

已完成的 VisionOPD Qwen3.5-4B：

- BLINK 62.65
- MMStar 70.93
- MMBench_DEV_EN 81.79
- VStarBench 86.91
- MathVista_MINI 79.20
- HRBench4K 82.38
- HRBench8K 干净重跑为 72.0
- 7-bench 76.55
- POPE 89.17
- HallusionBench 71.08 / 51.73 / 49.23
- ZoomBench 59.05

注意：

- 这个结果主要说明 Qwen3.5 底座更强；不能拿它和 Qwen3-VL contrast 直接证明机制差异。
- Qwen3.5 contrast-标准/保守训练仍在排队或进行中，尚不能下同底座结论。
- Qwen3.5 HRBench8K 曾有 56/800 API 失败污染，原始 73.00 不可引用；干净重跑 72.0 应作为当前数字。

## 消融与机制证据

### control 信号互补性

同一批 qvis rollout 离线重算四种 control 的 `ra_raw/ra_weight`：

| Pair | Jaccard | raw corr | weight corr | 解读 |
|---|---:|---:|---:|---|
| noimg-black | 0.633 | 0.721 | 0.685 | 最接近，都测视觉内容存在性 |
| noimg-degrade | 0.468 | 0.218 | 0.323 | degrade 提供细节/acuity 轴 |
| qvis-noimg | 0.488 | 0.100 | 0.148 | qvis 与视觉依赖信号弱相关 |
| qvis-black | 0.571 | 0.056 | 0.112 | qvis raw gap 与 black 基本不相关 |

可写：

- 不同 control 并非冗余；noimg/black 接近，degrade 更偏视觉细节，qvis 更偏问题依赖。
- 新排队的 qtext 实验就是顺着这个逻辑：保留真实图像、把问题换成 “What is the answer?”，用于做 case-level 互补性分析，而不是单纯比总分。

### 早期 noimg vs 纯 EMA

三组件分解中已有关键数字：

| 变体 | 7-bench 平均 | 相对 base |
|---|---:|---:|
| base | 65.78 | - |
| 纯 EMA 均匀权重 | 67.79 | +2.01pp |
| noimg 完整 | 68.20 | +2.42pp |

写法：

- 早期 noimg 的大部分收益来自 EMA self-teacher 正则化本身：`2.01 / 2.42 ≈ 83%`。
- token 级视觉相关性权重在早期 noimg 框架中只贡献约 +0.41pp；这正是后续需要 contrast target 的动机。
- shuffle 组状态在材料里仍有“评测跑中/待补”的痕迹；若最终数字未确认，不要把 shuffle 当完成结论写进主文。

### 权重/散度 Stage 1

在 noimg 上只换 weighting/divergence：

| 变体 | 8-bench 旧口径平均 | 7-bench 结论 |
|---|---:|---|
| noimg continuous forward | 64.32 | 最稳基准 |
| vaopd-grouped-forward | 63.75 | 轻微退步，大多在噪声内 |
| vaopd-grouped-rollout-forward | 62.53 | 真实、全面退步 |
| continuous-reverse | 64.04 | 与 noimg 基本无法区分 |
| continuous-jsd | 63.59 | 有 trade-off，VStarBench 提升但 MMStar/MMBench 退步 |

噪声本底来自 noimg 换训练 seed：

- 8-bench 平均只降 -0.38pp。
- 单项噪声不小：BLINK -2.26pp，HRBench8K -1.38pp，HRBench4K -1.25pp。

写法：

- 不应过度解释 0.3-1pp 的单项差异；需要参照 seed 噪声。
- `vaopd_grouped + rollout_reweight` 是唯一多数 benchmark 超出噪声本底的明确负结果。

### 外层 token 权重消融

uniform-weight（去外层权重）重跑修正后：

- 7-bench 68.97 vs contrast-标准 70.65，下降 1.68pp。
- POPE 88.97 vs 89.13 基本持平。
- Hallusion aAcc 69.40 甚至略高于 contrast-标准 69.19，但 fAcc/qAcc 43.35/45.05 低于标准 46.53/47.47。

可写：

- contrast target 是核心，但外层 token 权重不是可有可无；它主要贡献通用感知 benchmark，而不是 POPE/Hallusion aAcc。
- 这比早期 noimg 的“权重只贡献 +0.41pp”更强，说明 target 和权重在 contrast 机制中有协同。

### GRPO 对照

2B 本仓库数据：

- contrast-标准 7-bench 70.65，ZoomBench 43.67。
- GRPO step585/3ep 7-bench 70.19，ZoomBench 47.81。

2B virl39k：

- GRPO step464/1ep：7-bench 71.66，ZoomBench 38.34。
- GRPO step1392/3ep：7-bench 71.93，ZoomBench 41.18。
- contrast-标准 virl39k 90step：7-bench 70.68，ZoomBench 41.89。

4B Qwen3-VL：

- GRPO step585：7-bench 77.57，ZoomBench 53.25。
- contrast-保守：7-bench 77.13，ZoomBench 43.79。
- contrast-标准：7-bench 76.79，ZoomBench 44.62。

写法：

- GRPO 是强 baseline，尤其在可验证 MCQ/reward 数据上收敛快、ZoomBench 强。
- RA-VAD/contrast 的差异化贡献不能写成“打败 GRPO”，应写成“不依赖 verifiable reward，在无 reward 的蒸馏设定下达到接近的通用感知性能”。
- 如果下游任务没有干净 reward，GRPO 不可直接套用；contrast/RA-VAD 可作为 reward-free alternative。

## 评测口径与完整性注意

### ZoomBench judge bug

早期 `compare_vaopd_0701.md` 中记录了 ZoomBench native pipeline 的三个 judge bug，后续以 v3-fixed/canonical 为主：

1. `MCQ_BENCHMARKS` 漏掉 `zoombench`，导致 620 道 MCQ 被送入不稳定 LLM judge；人工核验显示规则分歧中规则判定可靠。
2. `extract_first_option()` 抓第一个大写字母而不是最后答案字母，长 CoT 会被误判。
3. 数字题长 CoT 提取失败，特别误伤长回答模型；noimg-4B 在 194 条被 LLM 判错的数字题中有 37 条其实数字和 GT 一致。

写法：

- 所有 ZoomBench 数字必须标明 pipeline。旧官方 pipeline 和 v3-fixed/native/canonical 不应混算。
- noimg/contrast 回答更长时，旧 judge 会系统性低估；这既是方法结果的口径风险，也是评测完整性贡献的一部分。

### API 失败污染

`eval_integrity_registry.md` 扫描 668 个评测输出，发现 13 个被 `"Failed to obtain answer via API."` 或空预测污染，污染数字不可引用，需要重跑推理而不是只重跑 judge。

重点污染项：

- contrast-conservative/standard 4B 的 HRBench4K/8K 旧结果有 12-16 条失败，报告 HTML 已给出修正后数字：标准 4B 7-bench 76.79，保守 4B 77.13。
- uniform-weight HRBench8K 原有 61/800 API 失败；修正后 HRBench8K 69.25，7-bench 68.97。
- Qwen3.5 VisionOPD HRBench8K 原有 56/800 API 失败；干净重跑为 72.0，不是原 73.00，也不是估算 78.5。
- `sr1filtered_step200...CORRUPTED_PORT_COLLISION` 目录有 871/1901 BLINK 失败，是端口冲突旧案，不可引用。

论文引用前动作：

1. 查 `eval_integrity_registry.md` 是否为 ✅。
2. 对 HRBench4K/8K 尤其谨慎，因为高分辨率大图最容易 vLLM 请求失败。
3. 并行评测要确保 vLLM 端口唯一；端口撞车会产生看似正常但严重污染的数字。

### HRBench 官方口径 vs VLMEvalKit

官方 Vision-OPD 数字复现显示：

- 官方 checkpoint + 官方 eval code：HRBench4K 81.25 vs 官方 81.50，HRBench8K 77.50 vs 官方 77.00，V*Bench 85.34 vs 官方 84.82，基本吻合。
- ZoomBench 复刻 48.52 vs 官方 53.49，约 -5pp 偏移来自 judge 模型差异；相对差距仍保持。
- 同一官方口径下 contrast-标准/保守 4B：V* 83.77/84.82，HRBench4K 81.12/79.38，HRBench8K 77.00/77.62，ZoomBench 43.08/42.01。

根因定位：

- VLMEvalKit 数字比官方 HRBench 低约 5pp 的主因是 prompt 风格，不是图像处理或采样参数。
- 官方 prompt “Select from the following choices.” 诱导模型先推理再作答，中位回答长度 262 字符。
- VLMEvalKit MCQ 模板 “Please select the correct answer from the options above” 诱导直接输出选项字母，中位 15 字符。
- 同 checkpoint 同 800 题，两边答案字母仅 78.9% 一致；CoT-then-answer 对高分辨率感知题稳定高约 5pp。

写法：

- 跨论文比较 HRBench 时必须声明 prompt 风格。
- 如果主表使用 VLMEvalKit 口径，就不要直接拿官方 HRBench 数字作同表横比；可放 appendix/analysis 说明 prompt effect。

## Caveats / Limitations

### 训练长度安全区

contrast target 存在自增强失控风险：

- 90 步健康，7-bench 约 69-71。
- 437+ 步长训崩溃，7-bench 58-64 或更低，表现为无限复读和中英混杂。
- 崩溃机制假设：target 相对 EMA teacher 现算，student 更新后 teacher 滑动追随，contrast target 再基于 teacher 继续锐化，形成 positive feedback。
- 真实边界在 90 到 437 步之间，150-200 步附近可能开始失控，但缺少更细扫描。

论文中应避免：

- 不要写“训练到收敛”；应写“短程训练 / early-stop regime”。
- 不要写“90 步最优”；只能写“90 步在安全区内”。

### 任务类型副作用

`experiment_settings` 中记录：MathVista +7.7pp 的同时，WeMath Strict -2.4pp，且超长回答占比 32.9% vs base 7.6%。说明 contrast target 可能鼓励更长、更解释性的输出，对严格多步数学推理或严格格式任务有副作用。

建议写法：

- “The method improves broad visual perception benchmarks but can induce verbosity and hurt strict reasoning-format metrics.”
- 中文可写：“该方法强化了视觉条件下的解释性分布，但也可能带来输出变长和严格判分任务退化。”

### Zoom/Search 类任务

4B 上 VisionOPD/GRPO 的 ZoomBench 优势真实存在：

- VisionOPD 4B 52.90，GRPO 4B 53.25。
- contrast-标准/保守 4B 44.62/43.79。

2B 上 contrast-标准 43.67 高于 baseline 42.49，也高于 noimg 36.45；但 4B 上没有超过 baseline 的 v3-fixed 44.14 太多，且远低于 VisionOPD/GRPO。

写法：

- 不要声称 contrast 解决了 zoom/search。
- 可写成“contrast target improves general visual perception and partially recovers search performance at 2B, but bbox-crop teacher remains superior on explicit zoom/search benchmarks.”

### 计算与规模限制

RA-VAD/contrast 要求 full-vocabulary distillation。Qwen3.5-4B 的 vocab 约 248K，全词表 logits 开销极大，Ray object store 溢出到磁盘，单步可达约 1000 秒。这个不是配置错误，而是机制硬约束。

可写进 limitation：

- 当前实现计算/显存开销高，尤其大词表模型。
- 未来方向是 top-k 近似、sampled softmax、分块 KL 或只对 plausibility support 传输/反传。
- 但当前代码中 RA-VAD 传非 null `distillation_topk` 会直接抛错，因此已有结果不能声称使用了 top-k 省算力。

### 数据与长度口径

- virl39k 90-step 用单图过滤子集 14,002 条，不是完整 ViRL39K。
- sr1 90-step 使用 `MAX_PROMPT_LENGTH=4096`，virl39k 使用 6144；长度差异可能带来约 1pp 量级影响，方向不定。
- GRPO virl39k 可用原始多图 filtered 版，因为没有全词表蒸馏 OOM；contrast 不能直接用多图版。

## 可放进论文的 claim 草案

### 强 claim

- Contrast-sharpened self-distillation substantially improves reward-free multimodal fine-tuning: Qwen3-VL-2B 7-bench from 66.05 to 70.65 on in-domain data, and to 70.68 on ViRL39K-filtered 90-step training.
- The contrast target is necessary beyond scalar token reweighting: early noimg reaches 68.20, while contrast-standard reaches 70.65.
- The outer token weights remain useful even after target sharpening: removing them drops 7-bench from 70.65 to 68.97.
- The method transfers to external visual reasoning data under controlled training length: virl39k 90-step reaches 70.68 and VStarBench 78.01.

### 中等 claim

- EMA self-distillation explains much of early RA-VAD’s improvement, but contrast target recovers a larger visual-conditioning benefit.
- Conservative vs standard settings show a stability/strength trade-off; conservative may peak earlier, standard may benefit more from 90 steps on virl39k.
- On 4B, contrast improves general 7-bench and HallusionBench over baseline/VisionOPD, but not ZoomBench.

### 不建议写成强 claim

- “优于 GRPO”：不成立。GRPO 在 ZoomBench 和部分 4B 结果上更强。
- “优于 VisionOPD”：只在特定 7-bench/VLMEvalKit 口径成立；ZoomBench/search 类 VisionOPD 仍强。
- “训练稳定收敛”：不成立。长训会崩。
- “90 步最优”：不成立，只是安全。
- “无需 token 权重”：不成立，uniform-weight 下降 1.68pp。

## 缺口与待补实验

1. **shuffle 权重对照的最终数字**：早期文档中它是核心消融，但材料里仍有待补/评测跑中的痕迹。若最终没有可靠数字，主文不要把“视觉相关性 vs 泛加权”写死。
2. **Qwen3.5 同底座 contrast 对照**：目前只有 Qwen3.5 VisionOPD 完整结果；contrast 标准/保守仍未形成同底座表。
3. **训练长度边界扫描**：90 健康、437 崩；缺 120/150/180/220 等细粒度点来定位失控开始。
4. **WeMath/严格推理扩展评测**：已有负信号，但需要更系统地量化 verbosity、格式错误、strict score。
5. **case-level qtext vs black 互补性**：已排队的新实验应输出四象限分析，而不是只看平均分。
6. **评测完整性重扫**：重跑污染项后应再次跑 `scripts/scan_eval_integrity.py`，更新 registry，再冻结论文表格。
7. **官方口径 appendix**：如果要和 Vision-OPD 官方数字对比，应单独放官方 prompt/eval code 表，不要混入 VLMEvalKit 主表。

## 建议论文结构

1. Introduction：reward-free multimodal self-distillation；视觉条件敏感度作为训练信号；contrast target 比 scalar weighting 更有效。
2. Related Work：Vision-OPD/bbox-crop teacher、VA-OPD/external teacher、GRPO/verifiable reward、contrastive decoding/self-distillation。
3. Method：
   - EMA teacher hi/control teacher-forcing。
   - RA weight pipeline。
   - contrast-sharpened target。
   - full-vocabulary weighted forward KL。
4. Experiments：
   - 2B main table：baseline / VisionOPD / noimg / GRPO / contrast conservative / contrast standard / uniform-weight。
   - External data 90-step table。
   - 4B scale table。
5. Ablation：
   - pure EMA vs noimg。
   - noimg weighting/divergence stage。
   - uniform weight。
   - step30/60/90 training length。
6. Analysis：
   - control signal complementarity。
   - prompt/eval pipeline sensitivity。
   - ZoomBench judge fixes and answer-length bias。
   - data distribution vs search ability。
7. Limitations：
   - long-training instability。
   - compute cost of full-vocab KL。
   - strict reasoning/verbosity side effect。
   - reward-based GRPO and bbox-crop VisionOPD remain stronger on some search/zoom settings。

## 最后引用口径清单

写主表前建议冻结以下口径：

- 7-bench：VLMEvalKit server, temp0, top_p0.8, top_k20, presence_penalty1.5, max_new_tokens4096。
- POPE/Hallusion：同 VLMEvalKit server；Hallusion 必须同时报 aAcc/fAcc/qAcc，外部比较优先看 qAcc。
- ZoomBench：canonical/native GPT judge，明确是否 v3-fixed；旧官方 pipeline 只作历史参照。
- HRBench 官方对比：单独标注官方 prompt/eval code，不能与 VLMEvalKit 主表直接混排。
- 所有 xlsx/csv 输出：必须 registry ✅，否则不可引用。
