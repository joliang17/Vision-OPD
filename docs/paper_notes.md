# Paper 素材汇总 — RA-VAD 对比锐化自蒸馏

> 创建：2026-07-15。目的：把散落在各文档/报告里"写论文会用到"的方法、结果、消融、caveat 集中到一处，
> 供后续讨论 paper 结构 / 数据集选择 / 补实验时直接引用。
> 主要来源：`docs/reports/ra_vad_contrast_report.html`（结果主报告）、
> `docs/experiment_settings_contrast_standard_virl39k_90step.md`（供论文用的设置详录，含逐行核对过的公式）、
> `docs/compare_vaopd_0701.md`（Phase 0-4 全过程）、`docs/queued_experiments_20260713.md`（进行中实验）、
> `docs/eval_integrity_registry.md`（评测完整性）。
> **本文件只做汇总与索引，数字若与上述来源冲突，以来源为准并回来修这里。**
> 2026-07-16：已合并 Codex 独立生成版（`paper_notes_codex.md`，留作参照）中的互补内容：
> EMA 收益分解、verbosity 副作用、control 互补性定量表、Stage 1 消融、复现风险、口径冻结清单、禁写 claim 列表。

---

## 1. 一句话贡献（candidate claims）

1. **核心方法**：纯自蒸馏（EMA teacher，无外部 teacher / 无 GT 标签 / 无 RL reward）框架下，
   把蒸馏 target 从纯 EMA hi 分布换成**对比锐化分布**
   `target = softmax(log p_hi + α·(log p_hi − log p_ctrl))`（限制在 β-plausibility 支撑集内），
   等价于把推理期 contrastive decoding 的几何倾斜 `p_hi^(1+α)/p_ctrl^α` 蒸馏进训练。
2. **诊断发现（motivation，两环证据链）**：
   - 原始设计把 `log p_hi − log p_ctrl` 只当标量权重用，但 EMA teacher 与 student 几乎处处一致——
     权重最高十分位上 teacher-student KL 仅 **0.0007 nats**，比同批 token 的 teacher 自身 hi-vs-ctrl
     差距小 **~1800 倍**。即"在给一个几乎不存在的信号做加权"。
   - 收益分解印证（compare 文档 Phase 2）：纯 EMA 均匀权重 7-bench **67.79** vs noimg 完整 **68.20**
     （base 65.78）——早期 noimg 收益的 **~83% 来自 EMA 自蒸馏正则化本身**，token 级视觉权重只贡献
     +0.41pp。→ 这正是"必须把对比信息写进 target 本身"的动机。paper 里可作 analysis 小节。
3. **首个正结果**：contrast-标准（2B, 62 步）是第一个在 ZoomBench 上超过原始 base 的 RA-VAD 变体
   （43.67 vs 42.49；此前 noimg/black/degrade/qvis/visionopd 全部低于 base 4-6pp），也是第一个在
   HallusionBench aAcc 上超过 base 的变体（69.19 vs 68.35）。注意：ZoomBench 上并未超过
   GRPO(本仓库 47.81)，不要写成"同时超 base 和 GRPO"。
4. **消融支持**：外层逐 token 权重 w_t 是必要成分（uniform-weight 掉 1.68pp，超出 ±0.38pp 种子噪声带）；
   训练长度存在安全区（90 步健康、437 步崩溃，机制是 target 自我放大的反馈环）。

## 2. 方法核心（正式写法素材）

- **训练范式**：on-policy 自蒸馏。student 采样 rollout → EMA teacher（`θ_T ← 0.95·θ_T + 0.05·θ_S`）
  在两个条件下 teacher-forcing 打分：hi=真实图像，ctrl=**同尺寸黑图**（消除"有没有图"元信号混杂）。
- **Loss**：`L = Σ_t w_t · KL(target_t ‖ p_student,t) / Σ_t w_t`，全词表 **forward KL**（teacher/target
  在前——`F.kl_div(student_logprobs, target, log_target=True)`，2026-07-16 对照 ra_vad.py:202 核实；
  报告 HTML 的 `KL(p_student ‖ p_hi)` 记法方向反了，勿沿用），温度 T=2.0（Hinton KD，loss 乘 T² 校正）。
  `reward_model.enable=False` 贯穿始终。归一化是 **batch 级**（`Σ_{b,t} w·KL / Σ_{b,t} w`，
  `denom=weights.sum()`，ra_vad.py:488），不是逐序列。梯度只进 student；teacher 仅走 EMA
  （dp_actor.py:146），target/权重全 detach。
- **⚠️ rollout IS 修正对 RA-VAD 路径无效（2026-07-16 Codex 审计发现）**：run 脚本配置了
  `rollout_correction.rollout_is=token, threshold=2.0`，trainer 也算了权重，但 `ra_kd_loss` 的调用
  （dp_actor.py:1348）**不接收** `rollout_is_weights`（非 RA fallback 路径有传）——RA-VAD 实际目标就是
  未修正的纯加权蒸馏。settings 详录 §3 那行"rollout 重要性修正"对 RA-VAD run 是配置了但不生效，
  论文 reproducibility 里不要 claim 做了 IS 修正。详见 overleaf-paper/AUDIT_FINDINGS.md。
- **对比 target 四重保护（guarded tilting）**：① softmax 保证合法分布；② β-plausibility 掩码（β=0.1，
  支撑集限制在 `p_hi ≥ β·max p_hi`）——**出处 = CD 论文（Li et al. 2023）的 adaptive plausibility
  constraint 移植**（CD 原文典型值也是 0.1），写作时引 CD，不是自创组件；③ exclude_token_ids
  （im_end 151645 / endoftext 151643）——精确语义是 **"tilt 豁免"而非从 target 删除**：这两个 token
  保留原 lp_hi 打分、不参与倾斜，但仍受 plausibility mask 约束（来自 settings 详录的逐行核对，比报告
  HTML 的"防止被拔高"表述更准）；④ 可选 position gate（主表 runs 关闭，效应归 α/gating 消融）。
  - **②③均无单独消融（2026-07-17 排验证）**：X13 零训练 precheck（β=0 与无豁免变体的 target 退化
    检查）先跑，P13/P14（真训练消融）视 X13 结果与"w_t claim 是否被去掉"决定——若 guarded tilting
    升格为核心机制，②③就是审稿人必问项。旧 precheck 只扫过 β∈{0.1,0.05}（两者几乎无差，
    argmax-change rate 差 <0.3pp），β=0 是空白。
- **w_t 五步流水线**（`compute_ra_weights()`, `verl/trainer/ppo/ra_vad.py:64-147`，全程 stopgrad）：
  raw 信号 `a_t = (lp_hi − lp_ctrl)_t` → relu(δ=0) → 逐行 95 分位裁剪 → 行内正值均值归一化 →
  样本级 sigmoid 门控 `g = σ(margin/0.5)·σ(ra_answer/0.1)`。
  - 组件归属（2026-07-17 与用户确认）：relu/clip/归一化/gate/n_min 都是**本仓库工程**，不是 VA-OPD
    原方案（VA-OPD 用 grouped 归一化，Stage 1 消融证明更差）；**gate 是否在 VA-OPD 有对应物待翻原文
    确认**（related work 写作前待办）。`min_positive_tokens=1` 是可证明的 no-op（relu 全零⇒权重本就
    全零），只作 implementation guard 表述、不消融；sample gate 是真实行为组件（所有 run 都开着），
    消融已排 P10（token权重+gate vs 仅token权重 vs uniform 三点分解）。
- **两个配置**：contrast-保守 α=0.5 + position gate 开；contrast-标准 α=1.0 + gate 关。
- attention-based raw signal（`attention_to_image_score()`）是代码预留，**从未在任何训练里启用**，
  报告结果全部走 logprob 差默认路径——paper 不应提及或只作 future work。
- 关键实现文件：`verl/trainer/ppo/ra_vad.py`（537 行）、`verl/workers/actor/dp_actor.py`。
  完整超参见 `docs/experiment_settings_contrast_standard_virl39k_90step.md`（已声明"供论文写作使用"，
  公式逐行核对过 + Codex 独立审查）。
- **复现风险声明（写 reproducibility 时要处理）**：① `ANSWER_VAL_TRAIN_FILE` 在特定 mtime/文件缺失
  条件下可能被 split 初始化逻辑静默重写——复现命令应显式禁用自动 split；② `resume_mode=auto` 是默认值，
  同名 `EXPERIMENT_NAME` 重跑会自动续训而非从头开始；③ black ctrl 应描述为"保留图像占位符、尺寸和
  视觉 token 数，只把内容替换为纯黑 RGB"，不要写成 no-image。

## 2.5 OPSD answer-hint baseline 的真实 loss 形态（2026-07-17 Codex 对照代码核实）

主表 OPSD 行（answer-hint baseline）实际跑的和 ours 有三处不同，写 Background/对比时要如实：
- **teacher 输入**：同一 prompt + **拼接 ground-truth 答案 hint**（`teacher_prompt_mode=answer_hint`，
  `ray_trainer.py:1233-1290`）；teacher 参数也是 EMA 模块（legacy teacher，dp_actor.py:135-157）——
  tex Background 已用脚注澄清"概念上的原始区别是 privileged answer-hint 输入"
- **loss**：走非 RA 的 `compute_self_distillation_loss`（dp_actor.py:1383-1400），
  **top-k=100 支撑集重归一化 + generalized JSD（divergence α=0.5）**（core_algos.py:1109-1165），
  无 token 权重、无 sample gate、无 ctrl 分支
- **ours 与之的差异**：全词表 forward KL + 对比锐化 target + w_t 流水线——所以 OPSD→ours 的对比
  是"配方整体替换"而非单因子；paper 的 Background→Method 过渡已按此写（先原始 loss，再两点改动）

## 3. 主结果数字（2B, Qwen3-VL-2B-Instruct）

评测口径：temperature=0, max_tokens=4096；7-bench+POPE/HallusionBench judge=gpt-5.4-mini-2026-03-17；
ZoomBench=canonical native pipeline + 外部 GPT judge。

### 本仓库训练数据（可直接横向比较的核心表）

| 模型 | 7-bench均 | POPE | HalluBench aAcc/qAcc | ZoomBench |
|---|---:|---:|---|---:|
| baseline（未训练） | 66.05 | 88.76 | 68.35 / 44.84 | 42.49 |
| VisionOPD (bbox-crop teacher, step65) | 66.20 | 未测 | 未测 | 37.51† |
| noimg（纯 hi target, step62） | 68.20 | 89.16 | 66.98 / 43.96 | 36.45† |
| GRPO（step585 / 3ep, ~9.4× 算力） | 70.19 | 87.64 | 66.14 / 43.30 | **47.81** |
| contrast-保守 (α=0.5, step62) | 69.07 | 88.41 | 68.56 / 45.93 | 40.59 |
| **contrast-标准 (α=1.0, step62)** | **70.65** | 89.13 | **69.19 / 47.47** | 43.67 |
| contrast-标准-uniform-w (消融) | 68.97⚠ | 88.97 | 69.40 / 45.05 | 补跑中 |
| OPSD answer-hint (trial301683547 step62, C1 干净重跑 2026-07-16) | **60.87**★ | 88.60 | 65.40 / 40.44 | 未跑 |

★ answer-hint 明细：BLINK 47.66 / MMStar 53.13 / MMBench 62.80 / V* 70.68 / MathVista 57.20 /
HR4K 68.75 / HR8K 65.88。0% API 失败（C1 重跑，早前"崩盘"结论因 20-30% 污染作废，但干净数字仍
**远低于 base 66.05**——naive OPSD+answer-hint 是真实退化，可作 diagnosis 叙事证据）。
⚠️ 与旧 step65 版 answer-hint 评测（~65.8，不同 trial 的另一次训练）差约 5pp，两次训练间差异
未归因（tokenizer 坑#7 修复前后/seed），引用前需决定用哪个口径并标注。

† = 旧 v3-fixed pipeline 数字，与同列 canonical judge 数字不可直接比。
⚠ = 2026-07-15 修正值（HRBench8K 61/800 条 API 失败重跑后 66.38→69.25，7-bench 68.56→68.97；
uniform-weight 差距 2.09→**1.68pp**，结论方向不变）。

### 单项亮点 / 弱点（"怎么读这张表"节选）

- **HallusionBench**：contrast-标准是首个超 base 的变体（+0.84 aAcc）——机制设计初衷兑现的维度。
- **MathVista_MINI**：单项最大提升，67.90 vs base 62.50（+5.4），超过 GRPO 用 9.4× 算力换来的 +5.5。
  → 引出数据消融问题：是机制能力还是训练数据偏 math？
- **MMBench_DEV_EN**：contrast 仍落后 base 约 1pp（77.06 vs 78.09），所有 RA-VAD 变体共有，
  paper 里要如实标出。
- MMStar 上保守(63.60) > 标准(62.53)，两配置互有胜负；标准版优势来自 BLINK/VStar/MathVista/HRBench。

### 4B（Qwen3-VL-4B-Instruct）

| 模型 | 7-bench均 | POPE | HalluBench aAcc/qAcc | ZoomBench |
|---|---:|---:|---|---:|
| baseline | 75.48 | 89.43 | 70.56 / 47.91 | 44.14¶ |
| VisionOPD (yijiangli step65, 已核实来源) | 74.40 | 89.60 | 70.45 / 49.67 | 52.90¶ |
| GRPO (step585/3ep) | **77.57** | 87.97 | 71.71 / 49.67 | **53.25** |
| contrast-保守 (step62, HRBench修正后) | 77.13 | 88.45 | **73.08 / 53.19** | 43.79 |
| contrast-标准 (step62, 修正后) | 76.79 | 88.71 | 73.19 / 52.75 | 44.62 |

¶ = 历史 v3-fixed 数字，GRPO/contrast 三行是 canonical judge，彼此可比。
- **4B 上保守 > 标准**（77.13 vs 76.79），与 2B 方向相反——α=1.0 在更大模型上不再"越强越好"，
  可作 scale-dependent 消融讨论点。
- 4B contrast 的 HallusionBench 全面最优（aAcc +2.5 / qAcc +5.3 vs base）——幻觉抑制是最一致的增益。
- GRPO 4B ZoomBench 明显领先；contrast 两变体低于 baseline（与 2B 模式一致）。

## 4. 数据消融（本仓库 vs virl39k-filtered vs sr1-filtered）

数据：virl39k-filtered = ViRL-39K 单图子集 14,002 条（geometry/chart math MCQ 为主）；
sr1-filtered = Vision-SR1 通用 VQA 14,355 条。

- **GRPO 换 virl39k**：7-bench 71.66（+1.47，只用 1/3 算力），但 ZoomBench 47.81→38.34（−9.5pp）
  —— GRPO 的 ZoomBench 优势强依赖训练数据分布，非机制自带。HalluBench 也随数据大幅变动
  （aAcc 66.14→68.45）。→ paper 论点素材："baseline 的优势项对数据分布敏感，contrast 更稳"。
- **contrast × 外部数据（90 步安全区）**：标准×virl39k 70.68 ≈ 本仓库 62 步 70.65；
  保守×virl39k 70.21；标准×sr1 69.65；保守×sr1 69.12。全部健康、无退化。
- **GRPO virl39k 3ep（compute-matched, step1392）**：7-bench 71.93 全场 2B 最高，ZoomBench 41.18。
  这是审稿人一定会问的 compute-matched 对照，数字已有。
- MathVista：换 math 数据后 contrast 是否再涨是核心问题——math suite 补测中（见 §7）。

## 4.5 其他消融与机制证据（合并自 Codex 版，2026-07-16）

**Control 信号互补性定量表**（Phase -0.5，同一批 qvis rollout 离线重算四种 control 的 ra_raw/ra_weight）：

| Pair | Jaccard | raw corr | weight corr | 解读 |
|---|---:|---:|---:|---|
| noimg–black | 0.633 | 0.721 | 0.685 | 最接近，都测视觉内容存在性 |
| noimg–degrade | 0.468 | 0.218 | 0.323 | degrade 提供细节/acuity 轴 |
| qvis–noimg | 0.488 | 0.100 | 0.148 | qvis 与视觉依赖信号弱相关 |
| qvis–black | 0.571 | 0.056 | 0.112 | qvis raw gap 与 black 基本不相关 |

→ 不同 control 并非冗余，可直接写进 analysis 小节。排队中的 qtext 实验（真实图 + 问题换成
"What is the answer?"）就是顺着这条线做 case-level 互补性分析。

**Stage 1 weighting/divergence 消融**（noimg 框架内只换加权/散度，8-bench 旧口径均分，
参照种子噪声：composite ±0.38pp，单项最高 −2.26pp）：

| 变体 | 8-bench 均 | 结论 |
|---|---:|---|
| noimg continuous forward | 64.32 | 最稳基准 |
| vaopd-grouped-forward | 63.75 | 轻微退步，多在噪声内 |
| vaopd-grouped-rollout-forward | 62.53 | 唯一超出噪声本底的明确负结果 |
| continuous-reverse | 64.04 | 与 noimg 无法区分 |
| continuous-jsd | 63.59 | trade-off：VStar 升、MMStar/MMBench 降 |

**Verbosity 副作用（重要 limitation，来自 settings 详录）**：contrast-标准×virl39k 在 MathVista
提升的同时 **WeMath Strict −2.4pp**，且**超长回答占比 32.9% vs base 7.6%**——contrast target 鼓励
更长、更解释性的输出，对严格多步推理/严格格式判分任务有副作用。建议写法："improves broad visual
perception but can induce verbosity and hurt strict reasoning-format metrics"。
（注：该处 MathVista 增幅详录记 +7.7pp，与报告主表口径 +4.5pp 不同，引用前核对是哪个对照/采样口径。）

**shuffle 权重对照警告**：早期文档里 shuffle（打乱 w_t 的 placebo 对照）是核心消融之一，但材料中
仍有"评测跑中/待补"痕迹——最终数字未确认前，主文不要把"视觉相关性 vs 任意加权"写死。

## 5. 训练动态 / 崩溃分析（可写成 analysis 小节）

- **反馈环机制**：target 相对当前 EMA teacher 现算，student 被推向"比 teacher 更极端"→ teacher EMA
  跟进 → target 继续加码。自我放大 → 300+ 步生成崩溃（中英混杂、无限复读），rollout-dump 确认崩溃
  始于 step 150–200，与 loss 回涨拐点吻合。GRPO 无此现象（target 是标量 reward，无分布漂移）。
- **验证**：437 步崩溃版截短到 90 步重训 → 7-bench 全部恢复正常（69.1–70.7）。安全边界在 90～437 步
  之间，未细分。
- **step30/60/90 中间点扫描**（2026-07-15 补齐）："90 步最好"不成立；准确表述是"90 步内完全健康、
  无随步数退化"。标准×virl39k 唯一单调升（69.62→70.68）；**两条"保守"线都在 step60 见顶后回落**
  （virl39k 70.43→70.21，sr1 69.67→69.12）——"保守配置更早触及天花板"是值得写的假设/后续消融。
- 旧 bbox-crop teacher（VisionOPD）的 loss 曲线在同一 y 轴上几乎不动——"没有比较信号可学"是这类
  self-distillation 的共同问题，不是个例。contrast 换 target 后第一次出现真实学习曲线。

## 6. 与 Vision-OPD 官方论文数字的对齐（Related/对比表素材）

官方 4B：V* 84.82 / ZoomBench 53.49 / HRBench4K 81.50 / HRBench8K 77.00。
- 官方 eval code + 官方 checkpoint 复现已完成（±0.5pp 吻合；ZoomBench −5pp 一致偏移=judge 差异：
  论文用 Qwen3-30B judge，复刻用 Azure GPT judge，相对差距保持）。
- 同一官方口径下我们的 contrast-4B：V* 83.77/84.82、HRBench4K 81.12/79.38、HRBench8K 77.00/77.62、
  ZoomBench 43.08/42.01 —— **HRBench/V* 与官方 VOPD 同档；ZoomBench 上 bbox-crop teacher 的优势真实存在**。
- **关键坑（写实验设置必须声明）**：本仓库 VLMEvalKit 比官方低 ~5pp（HRBench）的根因是 **prompt 风格**：
  官方 prompt 诱导 CoT-then-answer（回答中位数 262 字符），VLMEvalKit MCQ 模板诱导直接吐字母（15 字符），
  同 checkpoint 同 800 题答案字母仅 78.9% 一致，CoT 在高分辨率感知题上稳定 +~5pp。
  已排除图像预处理（16.7M 像素上限不缩图）和采样参数。**跨管线比较 HRBench 时必须声明 prompt 风格。**
- Qwen3.5-4B 新底座：visionopd 复现 V* 86.9 / ZoomBench 59.0 / MMStar 70.93。ZoomBench/HRBench4K/POPE
  基本对上官方；MMStar(−8.67)/HRBench8K(−8.4)/V*(−5.24) 系统性偏低，头号嫌疑=评测统一用的
  presence_penalty=1.5（为压制退化 ckpt 复读而设，对健康模型可能有害）——待做无 penalty 对照。
  contrast × Qwen3.5 还在训（全词表蒸馏 248K 词表，单步 ~1000s，是 RA-VAD 硬约束：
  `distillation_topk` 非 null 直接 ValueError）。

## 6.5 相关方法论文官方数字（baseline 锚点，2026-07-16 用户提供截图）

### VA-OPD 论文主表（Qwen3-VL-8B teacher → 2B student，Geo3K 训练）

| Method | WeMath | MathVista | MathVerse | Math Avg | HalluB | AI2D | MMMU | MMStar | OCRBench | Visual Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base (2B) | 36.8 | 63.9 | 19.6 | 40.1 | 52.6 | 77.8 | 49.1 | 56.1 | 85.9 | 64.3 |
| CoT-SFT | 39.7 | 64.4 | 23.9 | 42.7 | 51.0 | 75.7 | 48.7 | 57.1 | 86.2 | 63.7 |
| Off-policy KD | 38.9 | 65.3 | 24.3 | 42.8 | 52.3 | 76.0 | 49.3 | 57.4 | 85.5 | 64.1 |
| GRPO | 44.1 | 64.6 | 24.9 | 44.5 | 54.0 | 77.5 | **52.9** | 57.8 | 84.8 | 65.4 |
| PAPO | 44.8 | 64.5 | 25.8 | 45.0 | 54.0 | 77.8 | 52.7 | 58.0 | 84.7 | 65.4 |
| OPD | 43.3 | 63.7 | 29.1 | 45.4 | 52.0 | 75.8 | 50.9 | 59.7 | 84.7 | 64.6 |
| **VA-OPD** | **46.6** | **66.4** | **31.9** | **48.3** (+2.9) | **54.5** | **78.2** | 51.5 | **59.9** | **86.4** | **66.1** (+1.5) |

### VA-OPD Table 2（四配置，avg@8，Qwen3-VL 家族）

| T→S | Data | Method | Math Avg | Visual Avg |
|---|---|---|---:|---:|
| 4B→2B | Geo3K | OPD / VA-OPD | 45.3 / **47.4** (+2.1) | 64.4 / **65.2** (+0.8) |
| 8B→2B | Geo3K | OPD / VA-OPD | 45.4 / **48.3** (+2.9) | 64.6 / **66.1** (+1.5) |
| 32B→2B | Geo3K | OPD / VA-OPD | 51.1 / **54.8** (+3.7) | 65.3 / **67.3** (+2.0) |
| **8B→2B** | **ViRL39K** | OPD / VA-OPD | 46.4 / **50.2** (+3.8) | 65.5 / **68.0** (+2.5) |

8B→2B ViRL39K 单项（与我们 virl39k 实验最可比的一行）：
- OPD：WeMath 46.1 / MathVista 63.3 / MathVerse 29.7 ‖ HalluB 53.6 / AI2D 75.4 / MMMU 51.4 / MMStar 61.8 / OCRBench 85.2
- VA-OPD：WeMath 50.3 / MathVista 67.5 / MathVerse 32.7 ‖ HalluB 55.8 / AI2D 79.5 / MMMU 53.2 / MMStar 64.0 / OCRBench 87.5

### Vision-OPD 论文官方数字（visionopd 6k 训练，与 §6 一致，补 MME-RW 两列）

| Method | Param | V*Bench | ZoomBench | HRBench4K | HRBench8K | MME-RW | MME-RW-CN |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-Instruct | 4B | 81.68 | 44.97 | 78.50 | 76.25 | 63.27 | 62.92 |
| Vision-OPD-Qwen3 | 4B | 84.82 | 53.49 | 81.50 | 77.00 | 68.02 | 68.89 |
| Qwen3-VL-Instruct | 8B | 84.82 | 42.96 | 79.63 | 75.25 | 63.19 | 64.61 |
| Vision-OPD-Qwen3 | 8B | 89.53 | 53.02 | 83.00 | 79.50 | 67.03 | 68.96 |

### 口径 caveat（引用这两篇数字前必读）

- **VA-OPD 是 avg@8**（8 次采样平均），我们主口径是 temp=0 单次——不能直接同表混排，只能"引用式对比"。
- VA-OPD 的 HalluB 口径疑似 = 三指标平均：其 2B base 52.6 ≈ 我们 baseline 的 (aAcc+fAcc+qAcc)/3 = 51.6
  （报告 HTML 里的"51.60"）。对上了量级但差 1pp，需最终确认（这也回应了 §6 尾注里"52.6 对应哪个口径"
  的悬案——大概率是 3-均值而非单指标）。
- VA-OPD 的 MathVista 大概率是 MINI（base 63.9 vs 我们 62.5，量级吻合）。
- **⚠️ VA-OPD 的 MathVerse base=19.6 与我们实测 base=30.48 差 11pp，绝对不可直接比**（2026-07-16）。
  我们 VLMEvalKit temp0 口径下 2B base 的 5-split 均值 30.48（Text Dominant 40.5 … Vision Only 23.2；
  官方 19.6 反而接近我们的 Vision Only 单 split）。三个候选解释：avg@8 采样口径系统性压低 base
  （base 输出不稳定，采样平均吃亏；蒸馏后模型更确定性→avg@8 吃亏少，会**放大表观增益**）、判分提取
  实现不同、split 取法不同。含义：**VA-OPD 报的 MathVerse +12.3（19.6→31.9）与我们的 +0.8 不可比较
  ——他们的终点 31.9 ≈ 我们的 base 30.48**，无法排除"增益部分来自修复被压低的 baseline"。同类现象
  已有两例（WeMath 36.8 对不上两个口径、HRBench 官方差 5pp=prompt 风格），跨 harness 数字一律只做
  引用式并排、不做同表 Δ。
- **MathVerse 我方 +0.8 的 split 级真相**（2026-07-16，写 ablation 时可用）：不是无效果而是两头相消
  ——Vision Only +2.7 / Text Lite +3.2（读图改善，contrast 信号起效处）vs Text Dominant −2.3（纯多步
  推导，蒸馏不创造推理能力）。与"感知锚定窄增益"的 math suite 判读一致。
- **两篇 baseline 的关键区别**：VA-OPD 需要外部更强 teacher（4B/8B/32B→2B）；Vision-OPD 需要
  bbox-crop 构造 teacher 视图。我们的方法两者都不需要（纯自蒸馏）——这是定位差异的核心。
- 与我们已有可比数字（temp=0 口径，不同 harness，仅参考）：contrast-标准×virl39k-90step 的
  MMStar 63.87 vs VA-OPD(ViRL39K) 64.0；HalluB 3-均值 52.93 vs 55.8；MathVista 67.0 vs 67.5。

## 7. 缺口清单（写 paper 前需要补 / 需要决定的）

**结果缺口**
- [ ] uniform-weight 的 ZoomBench（补跑中）
- [ ] contrast × Qwen3.5-4B（标准/保守，训练中，trial 301683547 有被 kill 风险，301761390 已排接手）
- [ ] Qwen3.5-4B baseline 全量评测（v1 因残留 vLLM server 端口污染作废，v2 排队中）
- [ ] math suite（WeMath/MathVerse/MMMU/OCRBench）：contrast-标准×virl39k-step90 vs base，验证
  "MathVista 增益来自机制还是数据"——数据消融叙事的关键一块
- [ ] MathVista 复现差异待查：同 config 两次跑出 67.00 vs 69.20（疑 seed），引用前要定稿
- [ ] Qwen3.5 MMStar 无-presence_penalty 对照
- [ ] 任务4/5（Qwen3.5 baseline×virl39k、GRPO×virl39k）训练完成但 eval 未跑
- [ ] shuffle 权重对照的最终数字（见 §4.5 警告）
- [ ] **opsd(answer-hint) × virl39k 的 2B 训练+评测**（主表对手，§7.5 决策后新增的硬缺口；已确认
  checkpoints/ 无此目录，2026-07-16 已连同 P2-P4/E1-E5 一起排入 queue 文档顶部"Paper 主表/消融缺口"节）
- [ ] WeMath/严格推理类扩展评测：系统量化 verbosity、格式错误率、strict score（已有负信号，见 §4.5）

**方法学缺口（审稿人视角）**
- [ ] 全部单种子；已知 composite 种子噪声 ±0.38pp、单项最高 −2.26pp。至少核心对比（contrast-标准
  vs baseline vs GRPO, 2B）应考虑多种子
- [ ] 90–437 步之间的崩溃边界未细分（可作 limitation 或补一个 step150/200 点）
- [ ] α 只扫了 0.5/1.0 两点；position gate 与 α 是耦合变化的（保守=α0.5+gate，标准=α1.0+无gate），
  严格消融应解耦
- [ ] ctrl 构造只用了黑图（早期 Phase 有 noimg/black/degrade/qvis 四种 control 的结果可引用：
  见 compare_vaopd_0701.md，ZoomBench v3-fixed 上 qvis 38.11 > noimg 36.45 ≈ black/degrade 36.33）
- [ ] β、T=2.0、ra_clip_quantile=0.95 等次要超参无消融（可 limitation 一笔带过）

**口径注意（写作时的红线）**
- ZoomBench 列在各表里 pipeline 不统一（v3-fixed vs canonical GPT-judge），带 †/¶ 的数字不可混比
- HallusionBench 引用外部数字时用 qAcc（多数论文口径），不要用主列 aAcc
- 外部数据 contrast 系列训练 max_prompt_length 被 OOM 降级（virl39k 6144 / sr1 4096，默认 8192），
  数据口径有轻微差异（~1pp 量级），跨 len 对比要标注
- ZoomBench 早期自判分 pipeline 的数字一律不可用（已修复，现行口径=外部 GPT judge）
- 一切"Failed to obtain answer via API"污染问题查 `docs/eval_integrity_registry.md`；引用任何
  xlsx/csv 前确认 registry 状态为 ✅

**主表口径冻结清单（动笔前逐项锁定）**
- 7-bench：VLMEvalKit server 口径 = temp 0, top_p 0.8, top_k 20, presence_penalty 1.5,
  max_new_tokens 4096, judge gpt-5.4-mini-2026-03-17
- POPE/HallusionBench：同上；HallusionBench 必须同报 aAcc/fAcc/qAcc，外部比较用 qAcc
- ZoomBench：canonical native + 外部 GPT judge，标明是否 v3-fixed；旧官方 pipeline 只作历史参照
- HRBench 与官方对比：单独一张表标注官方 prompt/eval code，不与 VLMEvalKit 主表混排

## 7.5 已定决策（2026-07-16 与用户讨论定稿；**晚间被 arXiv 版计划部分覆盖，见 §7.6**）

## 7.6 arXiv 版定稿（2026-07-16 晚，用户与 mentor 确定，优先级高于 §7.5 中冲突项）

- **主表**：virl39k-filtered × 三底座（Qwen3-VL-2B / Qwen3-VL-4B / Qwen3.5-4B）× 三行
  （base / OPSD answer-hint / **ours**）。
- **ours = contrast-标准（α=1.0，无 gate）**；保守（α=0.5+gate）降为 ablation 行。
  推荐依据：2B 上明确赢（70.65/70.68 vs 69.07/70.21，且 virl39k 上唯一单调升曲线）；
  4B repo 上保守领先 0.34pp 在 ±0.38 噪声带内不构成反证；方法叙述少一个 gate 组件。
- **保守暂不进 paper（2026-07-16 深夜用户决定）**：保守 = α+gate 双因子耦合，P7（α=0.5 无 gate）
  解耦跑完之前，paper 里不出现保守的任何数字、不做任何 α/gating claim（tex 的 alpha-gating
  小节已改为 stub+todo；guard④ position gate 只作机制描述）。P7 完成后：α 曲线 =
  {0.5 无gate, 1.0 无gate[, 2.0 无gate=P8]}，gate 效应 = 保守(0.5+gate) vs P7(0.5 无gate)。
- **GRPO 移出主表** → appendix 参照（compute-matched 数字已有，不补新跑）。
- **ZoomBench 暂停**：新 eval 一律 9-bench，已有 Zoom 数字保留但不进主表。
- **motivation 句式**："current VLMs ignore visual evidence"——配 M1 定量实验
  （base 真图 vs 黑图答案不变比例）。
- **Ablation/分析五件套**：①decoding 可视化（target 分布 decode 成人话 + qualitative 案例）
  ②不同底座 ③不同 α（P7 解耦 α=0.5 无 gate 是刚需，现有保守行是 α+gate 耦合）
  ④accuracy curve（step30/60/90）⑤最大 gap benchmark 的 OPSD vs ours 逐样本分析。
- **unfiltered virl39k → 条件切换主口径（2026-07-17 16:3x 用户拍板，覆盖此前"不跑"决策）**：
  验证门 Stage V（P15/P16 eval + P22 OPSD-unfiltered + P23 换seed）全过（标准：7-bench差≤0.5pp、
  无单项>2pp 系统性差）→ 主口径切到 unfiltered-1img（36,039/38,327，仅剔 6% 多图=机械性声明），
  Stage S（P24-P29）补齐主表+消融；filtered 结果降为 appendix 敏感性一行。动机：38.3K→14.8K
  过滤来历难考、解释成本高。已验证点：2B ours 两版 7-bench 完全打平（70.68=70.68）。
  详见 queue 文档 "UNFILTERED 切换计划" 节。**切换确认前 tex 不动。**
- **主表进度（2026-07-16 22:1x）**：2B 三行全齐——base 66.05 / OPSD **67.59**（E1：BLINK 55.29,
  MMStar 60.13, MMBench 73.11, V* 75.39, MathVista 62.1, HR4K 74.375, HR8K 72.75, POPE 88.58,
  HalluB aAcc 68.24, Zoom 38.46）/ ours 70.68。4B 缺 P5/P6 训练；Qwen3.5 缺 E7 eval 和 T3a 训练。
  注意：answer-hint×virl39k(67.59) **高于** base，与 repo 数据版(60.87 低于 base)方向相反——
  naive OPSD 的表现依赖训练数据，diagnosis 叙事写作时要区分口径。

**训练数据**：主线 = **ViRL39K**（公开可复现、compute-matched GRPO 对照已有、标准配置 step90 仍单调升）；
本仓库 6k 数据降级为 motivation/消融板块（诊断故事 + uniform-weight 原始版本 + "ZoomBench 首超 base"
claim 仅在此数据成立）。

**与 VA-OPD 的关系**：**不做数字对比**（setting 不同：它需要外部 8B/32B teacher；口径不同：avg@8）。
Related work 里做定位讨论：VA-OPD 把视觉依赖信号用于 teacher-based 蒸馏，我们用于纯自蒸馏——
"既不需要 teacher 也不需要 verifiable reward" 是监督成本光谱上的独立格子。
为避免"沉默对比"嫌疑，benchmark suite 不照搬它的 8-bench，用下面自建 suite。

**主表 benchmark suite（四板块 10 项，全部已有数据或在跑）**：

| 板块 | Benchmark | 理由 |
|---|---|---|
| ① 通用感知 | BLINK, MMStar | 机制主战场 |
| ② 幻觉/视觉接地 | POPE, HallusionBench（aAcc/fAcc/qAcc 全报） | 设计初衷指标，4B qAcc +5.3 最亮 |
| ③ 高分辨率/细粒度 | V*Bench, HRBench4K/8K, ZoomBench | Vision-OPD 血统板块，ZoomBench 如实报弱 |
| ④ 推理迁移 | MathVista_MINI（+可选 MathVerse_MINI） | 支撑"机制 vs 数据"消融 |

**Appendix（主动披露，不进主表）**：
- MMBench_DEV_EN：所有 RA-VAD 变体共有的 ~−1pp 现象，appendix 披露一行，避免"太常见却缺席"的质疑
- WeMath：verbosity 副作用板块（Strict −2.4pp、超长回答 32.9% vs 7.6%），写成 analysis 而非被动挨挖

**主表对手（全部同 setting、同 harness，2026-07-16 更新）**：Base / GRPO 1ep + compute-matched 3ep /
**opsd(answer-hint) 版本**。**Ablation 表**：纯 EMA / noimg / uniform-weight（+ step30/60/90、
保守 vs 标准）。VA-OPD、Vision-OPD 官方数字只在 related work / 单独引用表出现。
⚠️ 主表若锁定 virl39k：opsd(answer-hint) × virl39k 的 **2B** 训练+评测目前未见现成结果
（已有的是 repo 数据 2B step62 版和 Qwen3.5-4B×virl39k 版，后者 eval 也没跑）——需要补，已入缺口清单。

**取消的补实验**：Geo3K 训练 ✂️（价值绑定在 VA-OPD 锚点上，已不需要）、AI2D 评测 ✂️、OPD 8B→2B
内部复现 ✂️。**算力改投**：① uniform-weight × virl39k（消融搬家，主表一致性）；② virl39k
step120/150（崩溃边界扫描 + 可能的免费提升）；③ WeMath verbosity 缓解探索。

## 7.7 uniform-weight 反转事件与 w_t claim 的待定状态（2026-07-17）

- **E2 结果**：uniform-weight × virl39k (2B) 7-bench ≈ **71.01 > ours 70.68**（BLINK +3.3 / MMStar +1.4 /
  MMBench +2.3 / Zoom +1.7）——与 repo 数据的 −1.68pp **方向相反**，"w_t 必要"结论是数据依赖的。
- **用户决定**：P11（4B）/P12（Qwen3.5-4B）补 uniform-weight×virl39k 跨 scale 验证。
  判定规则：若 2B/4B/Qwen3.5 三点 uniform 都 ≥ ours−0.4pp，**method 正文去掉 w_t 权重 claim**
  （w_t 降为 inherited/optional 组件进附录）；若 scale 间分裂，如实写数据/scale 依赖。
- **写作暂缓项**：tex 里 ablation 关于 uniform −1.68pp 的段落（基于 repo 数据）在 P11/P12 出数前
  不要加强、也先不改——等三点齐了一次性重写。§1.4 的"消融支持"条目同此。
- eval 分工（2026-07-17 用户定）：所有 eval 由 mlx session 提交，清单在 queue 文档顶部 X1-X16。

**2026-07-17 15:5x 增量（E9/X系列回填后）**：
- **filter 敏感性已闭环（2B）**：unfiltered 7-bench 70.68 = filtered 70.68，单项 ±2pp 内——
  主表 filter 声明有实验背书。4B(P15)/Qwen3.5(P16) unfiltered 训完待评（2×2 表在路上）。
- **uniform 反转在 Qwen3.5 上干净复现**：X12(uniform)@4096 7-bench 79.89 vs X1(ours)@4096 78.95
  （+0.9，同 len）；2B +0.3（同 len 6144）。**4B 点被 len 混淆**：X11@4096 76.47 vs X5@6144 73.58
  的 +2.9 不能归因权重——P19（4B ours@4096）出数前 w_t 三点判定不能下。
- **⚠️ X5（4B ours@6144）7-bench 73.58 低于 4B base 75.48**（BLINK 61.65 vs 67.18）——4B 上加权
  ours 疑似伤模型；P19 分辨 len vs 权重的锅。
- **len 口径地图**：2B 全系 6144 一致✅；4B ours/OPSD 6144、uniform/unfiltered 4096⚠️；
  Qwen3.5 ours/uniform/P12 4096、OPSD(task4) 6144⚠️、unfiltered(P16) 6144（线性注意力，6144 稳过，
  "Qwen3.5 必须 4096"的旧规则对 unfiltered 不成立）。两处行内混淆由 P19/P20 修复，其余标注即可。
- 新增实测 variance 点：ZoomBench seedA/B ±1.3pp；X16（data.seed=1234，历史所有 run 数据顺序
  其实完全相同——`data.seed=null` 默认 generator 种子固定，重要发现）待评。
- Qwen3.5 主表行数字已出：ours(X1) 78.95 / OPSD(X2@6144) 76.66（7-bench）；4B：ours(X5) 73.58 /
  OPSD(X6) 71.30 / uniform(X11) 76.47——**4B 行在 P19 出数前不要填进 tex**。

## 7.75 候选终局配置（2026-07-17 21:2x 与用户讨论）：ours = 纯 contrast target（uniform，无 w_t 无 gate）× unfiltered

- 证据：uniform vs ours（weight+gate）同 len 比较——2B +0.3（噪声内）、Qwen3.5 +0.9、4B 等 P19 eval。
  表述纪律：只能写"权重不必要（打平）"，不写"uniform 更好"（差距在 ±0.38~1.3 实测噪声带量级）。
- 好处：①方法纯化为 contrast target + guarded tilting 单机制；②诊断叙事闭环（权重救不了→改 target→
  target 带上信号后权重连辅助价值都没有）；③与 unfiltered 切换汇合——最终主表 ours 行 = P26/P27/P28
  （uniform×unfiltered，正在训），X4/X3/X15 自动成为分解消融。
- 前置（未齐不动 tex）：P19 eval（4B w_t 判定门）+ P26/27/28 evals + Stage V 验证门。
- 附带：90 步 = 2,880 unique prompts（batch32×90，不放回）= filtered 20.6% / unfiltered 8.0%，
  写 setup 时如实声明（"short-horizon"定位的量化依据）。
- repo-6k 权重 +1.68 的旧点 → corpus-dependence 附录行保留。

## 7.76 guarded tilting 去留判定（2026-07-17 深夜）

- **EOS 豁免**：非 CD 标准（本仓库工程）——X14 打平即从 Method 删（代码可留作 implementation detail
  一句话，或 FINAL-WAVE 一并去掉）。
- **β-plausibility**：Codex 独立裁决 KEEP-AS-SAFEGUARD（数学：T=2 下 β=0 target ∝ p_hi/√p_ctrl 仍
  无界；forward KL mass-covering + EMA 环使训练期毒性累积），**但用户指出关键事实：437 崩溃就发生在
  β=0.1 开着时**——"β 拦自增强"已被证伪，β 仅存辩护 = "没有它崩得更早"。
- **判定实验**：P31（β=0 续到 200 步）vs P21（β=0.1 续到 200 步）两臂对比崩溃起点——150-200 区间
  同样健康 → **β 删**；更早恶化 → β 留（且有正面证据）。90 步打平（X17）不构成删的充分条件。
- **主表重跑策略**：不逐项重跑——所有判定（P31/X14/X17/P19/Stage V）齐后一次锁定终局配置
  （candidate：裸 tilt + uniform + unfiltered），只跑一波 FINAL-WAVE（3 训练+3 eval）。
  若 β/EOS 判定翻盘则保留该件、P26-28 现有产物直接可用。
- 叙事纪律：无论 β 去留，"我们精确蒸馏 CD"不作为核心卖点（我们是 OPSD，不是 decoding 方法）；
  β 若留，写成"防 forward-KL 软 target 被无界 ratio 污染的 safeguard"并引 CD 作约束出处。

## 7.77 三大判定收口 + 终局配置确认（2026-07-18，devbox 从 NAS 产物直接计算）

- **w_t 判定 = 删除 ✅**：三 scale 同 len 对比全过 §7.7 门——2B uniform 71.01 vs 70.68（+0.3）、
  **4B uniform 76.47 vs P19 ours@4096 76.68（−0.2，噪声内）**、Qwen3.5 79.89 vs 78.95（+0.9）。
- **4B 之谜 = len6144 的锅**：P19@4096 76.68 > base 75.48（X5@6144 73.58 的"低于 base"解除）；
  4B 主表行一律用 @4096 口径。
- **β 判定 = 保留 ✅（反转）**：X17（β=0，90步）7-bench 69.11 vs ours 70.68 = **−1.57pp 系统性掉分**
  （BLINK −2.2/MMStar −2.4/MMBench −3.2/MathVista −2.1，仅 Zoom 持平 41.89）——β 在 90 步内就
  load-bearing，不必等 P31。用户"加了也崩"论点仍成立（β 不防长程崩溃）——写作时 β 的角色 =
  "防短程质量损失的支撑集约束"，引 CD；崩溃另有机制（teacher 漂移）。P31 降级 nice-to-have。
- **终局主配置确认**：**uniform × unfiltered × β=0.1 × α=1.0（+ EOS 豁免待 X14 尾巴）** ——
  恰好 = P26/P27/P28 已训配置，FINAL-WAVE 改型为"续训到 150 + eval 90/120/150"（见 queue），
  不需要重训。
- **边界曲线（旧配置，单条）**：90/120/150 = 70.68/**71.36**/71.05，峰值 ~120——但 constant-lr
  schedule 绑定 + 单曲线 + 非终局配置，不可外推；max-step 由 FINAL-WAVE 三 scale 曲线定。
- Stage V 进度：V-e1 vs P19 = 77.25 vs 76.68（unfiltered **+0.57**，略超 ±0.5 门但方向有利）；
  V-e3 完成待算；V-e2 缺 POPE/HalluB；V-e4 缺 4 项。整体判定等这两个收口。

## 7.78 方案 B 确认 + seed 混淆纠正 + hint 依赖发现（2026-07-18 下午）

- **seed 混淆纠正**：昨晚"2B unfiltered 掉分"判定拿 V-e4(seed1234) 对比默认 seed 主表——错配。
  同 seed 配对后 2B ours filtered=unfiltered 打平（70.68/70.68 与 67.65/67.88）。
  **真发现 = 2B run-to-run variance ≈ ±3pp**（两语料一致），旧 ±0.38 门作废；
  uniform vs weighted 同 seed 配对后全部 <±1pp = 噪声内不可裁决。
- **用户拍板：维持方案 B（ours = uniform × unfiltered × β0.1 × α1.0），依据=故事简洁**（数据无法裁决时
  的合法标准；写作时如实说 "weights neither help nor hurt within run-to-run variance"）。
  FA1-4/P28/FINAL-WAVE 续训照跑；S1 seed 加固（主表 2B 行报 mean±std）。
- **V-e3 归因（S2a 审计完成）= hint 依赖幻觉，礼物级发现**：answer-hint×unfiltered 2B 的崩溃
  不是 infra——模型 eval 时凭空引用"reference answer"并朝想象答案推理（12.6% vs filtered 2.9%）。
  = privileged-hint OPSD 的 train-test mismatch 本征缺陷；黑图 ctrl 无信息可依赖、结构性免疫。
  可进 motivation/analysis；S2b 换 seed 复现确认中。
- **L1 探针排队**：4B ours@len6144 的低分可能是 eval max_new_tokens=4096 截断假象——@8192 重评
  两臂（X5 ckpt vs base）验证；若反超 base 则主表 eval 口径要重新考虑。

## 7.79 "为什么 90 步"的处理（2026-07-18，Codex 裁决 + 用户确认方向）

- **裁决：保持 90，不跑 100**（round number 无科学增益，重训纯化妆成本）。
- **写法**（Codex 起草，setup 直接用）：90 = 固定短程算力预算（2,880 unique prompts @bs32），
  预先选在 150-200 崩溃区之前；"We do not claim that 90 steps is optimal; instead, we report an
  accuracy-vs-steps analysis over later checkpoints to show plateau stability."
- **⚠️ 步数选择纪律（防 test-set selection）**：若 FINAL-WAVE 显示 120 跨 scale 且超 ±3pp 噪声地更好，
  **不得**以"全 suite argmax"为由把 headline 换成 120——只能 (a) 坚持预注册固定 90，或 (b) 用
  held-out 验证/单一预声明 dev benchmark 选步数、全 suite 只报一次。
- **FINAL-WAVE eval 报告要求**：90/120/150 × 三 scale 同口径；逐 benchmark + aggregate 双列；
  与 ±3pp run 噪声对比判 plateau；150 处的复读/混语定性检查一并报。
- **用户保留决策（2026-07-18 记录，同日晚修正）**：暂按"保持 90"执行，保留改判空间——之后视情况做
  细粒度曲线（每 10 步 eval）。**⚠️ 修正：中间 checkpoint 并非白拿**——prune 政策只留 30/60/90/120/150
  （已实核 P26/P27 目录），10 步粒度的中间点需**同配置重训一次**（save_freq=10 且暂缓 prune，2B ~2.5h）。
  且因 GPU 非确定性 ±3pp，**细曲线必须整条来自那次新 run（含它自己的 90/120/150），不得与原 run 的
  端点数字混拼**。届时按需登记为 FC1（fine-curve rerun）。换步数仍受 §7.79 纪律约束
  （held-out 选步或预注册，不得全 suite argmax）。

## 7.8 virl39k 主线最新数字整备（2026-07-17 16:0x，供 tex 更新用；全部 infer_fail 0%）

### 主表（三底座 × base/OPSD/ours，virl39k-filtered，90步，9-bench 无 Zoom）

| 行 | BLINK | MMStar | POPE | HalluB(aAcc) | V* | HR4K | HR8K | MathVista | **7-bench** | 状态 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2B base | 53.02 | 57.47 | 88.76 | 68.35 | 72.77 | 71.13 | 67.38 | 62.50 | 66.05 | ✅ |
| 2B OPSD | 55.29 | 60.13 | 88.58 | 68.24 | 75.39 | 74.38 | 72.75 | 62.10 | 67.59 | ✅ |
| 2B ours | 58.18 | 63.87 | 88.36 | 68.56 | 78.01 | 76.38 | 73.13 | 67.00 | 70.68 | ✅ |
| 4B base | 67.18 | 68.93 | 89.43 | 70.56 | 80.63 | 79.88 | 73.75 | 73.90 | 75.48 | ✅ |
| 4B OPSD (X6) | 57.55 | 64.67 | 87.45 | 66.77 | 79.06 | 76.00 | 72.37 | 66.10 | 71.30 | ⚠️ **暂缓进 tex** |
| 4B ours (X5@6144) | 61.65 | 67.13 | 87.82 | 70.56 | 83.25 | 76.62 | 73.88 | 69.70 | 73.58 | ⚠️ **暂缓进 tex**（低于 base！len 混淆，等 P19@4096） |
| Qwen3.5 base | 66.91 | 72.67 | 83.11 | 74.66 | 84.29 | 86.13 | 82.00 | 80.70 | 79.06 | ✅ |
| Qwen3.5 OPSD (X2@**6144**) | 64.86 | 72.13 | 88.70 | 74.34 | 84.29 | 81.25 | 77.75 | 77.90 | 76.66 | ✅ 可进（脚注 len6144，P20@4096 重训中拉齐） |
| Qwen3.5 ours (X1@4096) | 66.23 | 75.33 | 88.39 | 76.13 | 82.72 | 85.12 | 78.25 | 82.30 | 78.95 | ✅ 可进 |

**Qwen3.5 行叙事（小心措辞）**：naive OPSD 掉到 76.66（−2.4 vs base）；ours 恢复到 78.95 ≈ base
（−0.11，噪声内），且 POPE +5.3（88.39 vs 83.11）、HalluB +1.5、MMStar +2.7——"ours 在 7-bench 上
与 base 打平但幻觉/感知细项显著改善；naive OPSD 则明显退化"。不许写"ours 超过 Qwen3.5 base 综合分"。

### 消融/分析新数字

- **uniform-weight 三点（w_t 判定，§7.7）**：2B uniform 71.01 vs ours 70.68（**+0.3**，同 len6144）；
  Qwen3.5 uniform(X12) 79.89 vs ours 78.95（**+0.9**，同 len4096；X12 明细：68.12/75.53/86.00/83.25/
  81.7/84.62/80.00，POPE 88.44，HalluB 75.39，Zoom 53.49）；repo-2B 旧点 −1.68；
  **4B 点（X11 76.47 vs X5 73.58）len 混淆，等 P19**。→ tex 措辞："on the main corpus the outer
  weights do not appear necessary; final claim pending the length-matched 4B run"。
- **filter 敏感性（E9）**：2B unfiltered 70.68 = filtered 70.68，单项 ±2pp——setup/appendix 一句带过。
- **α/gating（X3/X4）**：仅 Zoom 出数（α0.5-nogate 43.43 / no-gate 41.07 vs ours 41.89，都不敏感）；
  9-bench 在跑——α/gating 小节维持 stub，只可加"Zoom 上不敏感"一句。
- **run-to-run variance 实测**：Qwen3.5 保守 seedA/B ZoomBench 50.89 vs 49.59（**±1.3pp**）；
  X16（data.seed=1234）待评。**重要方法学发现**：`data.seed=null` 时 generator 种子固定
  （67280421310721）⇒ 历史所有 run 数据顺序完全相同——reproducibility 声明可写，也解释了
  "同配置差异来自 GPU 非确定性而非数据顺序"。
- **X8（appendix verbosity 对照）**：answer-hint 2B 的 MathVerse_MINI 30.20 / WeMath Strict 32.29
  （Loose 49.81）。
- **P13/P14/P8/P16 训完待评**（guarded-tilting 消融、α=2.0、Qwen3.5 unfiltered）——tex 不动。

## 8. Paper 定位讨论的开放问题（留给后续讨论）

1. **主 claim 选哪个？** (a) "对比锐化 target 修复了自蒸馏无信号问题"（方法+诊断故事，最完整）；
   (b) "无 reward 自蒸馏逼近 GRPO 且幻觉更优、算力 1/9"（效率故事，但 7-bench 仍略输 compute-matched
   GRPO 71.93）；(c) 幻觉抑制角度（HalluBench 4B +5.3 qAcc 最亮眼且跨 scale 一致）。
2. ~~**主实验数据集用哪个？**~~ ✅ 已定（见 §7.5）：主表 virl39k，本仓库数据作 motivation/消融。
3. **base 模型矩阵**：Qwen3-VL 2B+4B 已齐；Qwen3.5-4B 是否等结果、是否值得为它承担全词表蒸馏的
   算力成本（单步 1000s）。
4. **ZoomBench 弱势怎么处理**：如实报 + 用"GRPO 的 ZoomBench 优势数据依赖（−9.5pp 换数据即崩）"
   对冲，还是补一个 zoom 类数据混训实验。
5. GRPO+RA-VAD 结合（compare 文档 Phase 2 提过）是否作为额外贡献点。

**禁写 claim 清单（不成立或证据不足，写作红线）**

- **训练 loss 曲线的形态（437 步回涨、爬升轨迹等）不入图、不在正文提及**（2026-07-17 用户指示："loss 长得不对头"）。步数安全区/崩溃结论只用 eval 分数与生成质量证据（复读/中英混杂扫描）表述，不引用 loss 走势。
- **L1/L2 长度探针（同一 checkpoint @4096 vs @8192 的 8 组配对，全部 ±0.6pp，"截断假设关闭"）——只作内部存档，不进 paper**（2026-07-19 用户指示）。总账 `ra_vad_results_ledger.html` 口径冻结区仍保留这条记录供内部核对 eval 口径正当性，但写作时不得引用或提及这批数字/结论。
- ❌ "优于 GRPO"——ZoomBench 和 4B 部分结果 GRPO 更强；compute-matched GRPO(71.93) 7-bench 也更高。
  正确写法："不依赖 verifiable reward，在无 reward 蒸馏设定下达到接近的通用感知性能"。
- ❌ "优于 VisionOPD"——只在 7-bench/VLMEvalKit 口径成立；ZoomBench/search 类 VisionOPD 仍明显更强。
- ❌ "训练稳定收敛"——长训会崩，只能写 "short-horizon / early-stop regime"。
- ❌ "90 步最优"——只是安全区内，最优点非单调。
- ❌ "无需 token 权重"——uniform-weight 掉 1.68pp。
- ❌ "解决了 zoom/search"——2B 上略超 base，4B 上远低于 VisionOPD/GRPO。

（Codex 版 `paper_notes_codex.md` 末尾另有一份七段式论文结构建议，可作动笔时的骨架参照，此处不重复。）

## 13. nothink vs thinking base 对照（Qwen3.5 系，2026-07-22 用户提供，暂存备用）

主表 base 用 **thinking** 口径（base 开 thinking）。以下是关掉 thinking 的 base（nothink）对照：

| 底座 | nothink base | thinking base（主表） | thinking 溢价 |
|---|--:|--:|--:|
| Qwen3.5-2B | 68.61 | 68.59 | ≈0（−0.02） |
| Qwen3.5-4B | 73.94 | 76.45 | +2.51 |
| Qwen3.5-9B | 74.97 | 78.68 | +3.71 |

**观察：thinking 溢价随 scale 增大**（2B≈0 / 4B +2.51 / 9B +3.71）——大模型 thinking 更有用、小模型几乎无差。**用途（可能，用户"之后可能用"）**：① no-think 公平对照（ours 训后天然 no-think，若 base 也报 no-think 对照更公平，类似 Qwen3-VL 的 no-think 对照叙事）；② thinking-scale 讨论点。**注意口径**：这三个 nothink 值需确认是否与主表同 suite（7-bench Average HR + Hallu 三均）；引用前核对。

## 12. 新 suite 消融整体 Acc（2026-07-22）

> ⚠️ **本节部分 HR 值是 cycle0 口径（devbox 早期误用），已作废。paper 一律以
> `paper_handoff_values_20260722.md`（301967423 出具，HRBench Average 口径、经核验）为准。**
> 口径三处：HR=`cycle=Average×type=all`；Hallu=aAcc/fAcc/qAcc 三均；α=1.0 全文统一 66.66。

口径：7-bench = BLINK/MMStar/V*/MathVista/HR4K/HR8K/Hallu三均；VLMEvalKit vllm_server，temp0/pp1.5/4096，judge gpt-5.4-mini。base(未训练)=62.27，ours(2B P26)=66.64。**paper 里 α/β 表按用户要求只报整体 Acc（不逐项）。**

### α 消融（对比强度，2B×unfiltered step90，uniform）
| α | Acc | 状态 |
|---|--:|---|
| 0 (matched, tilt 关) | 64.31 | ✅ 重判干净 |
| 0.5 (FA1) | 63.56 | ✅ 07-22 重判（仍略低于 α=0，疑单 seed 训练波动） |
| 0.75 (P37a) | — | 只 step10 未训完，待重训 |
| 1.0 (ours) | 66.64 | 主表 |
| 1.25 (P37b) | 67.04 | ✅ 07-22 判完 |
| 1.5 (P37c) | 66.14 | ✅ 07-22 判完 |
| 2.0 (FA2) | 64.32 | ✅ 07-22 重判干净（掉回≈α=0，单峰确认） |
对比 tilt 净贡献 = ours − α=0 = **+2.33pp**（matched baseline，隔离 reweight/自蒸馏）。α=0.5=63.05<α=0 的非单调是 FA1/FA2 被 verbose-deflation 压低的假象，待重判后填。

### β 消融（plausibility support 阈值，2B×unfiltered step90，uniform）
| β | Acc | 状态 |
|---|--:|---|
| 0 (无 support 限制, FA3) | 65.14 | 07-20 判，无降级字样 |
| 0.1 (ours, main) | 66.64 | 主表 |
β support 净贡献 = **+1.50pp**（逐项 6/7 正：BLINK+0.90/MMStar+0.27/V*+1.05/MathVista+1.80/HR4K−0.12/HR8K+1.62/Hallu**+4.96**）。gap 主要在 Hallu，比 anchor 明确。FA3/ours 待同口径重判定稿，方向稳。

**β=0 长训练崩溃（旧口径 Table2，2026-07-22 核实 step200 无 judge 降级字样=真崩，非坏 judge）**：β=0 单调崩溃 step90 **69.11**（Table2 现用 68.90）→ step150 **60.06**（Table2 用 59.52）→ step200 **57.72**（逐项 BLINK 43.4/MMStar 46.2/MMBench 59.3/MathVista 40.2 全崩，HR4K 73/HR8K 75.5 保留）；β=0.1 全程 70.68/71.05/70.36/70.99 平稳。→ 强证据：unconstrained support 下 recursive target distortion 逐步累积。**Table2 的 step200 pending 格可填 57.72**（step90/150 保持现有）。**step180 checkpoint 已被 prune**（ext200 dir 只剩 step30/60/90/110，无 180/200 的 ckpt；step200 的 eval 数据是崩溃前跑好的、仍在），**无法补 eval** → Table2 的 β=0 行改用 **step90/150/200 三点**展示崩溃即可，step180 列删除或留空。

### anchor 消融（no-anchor P33 vs ours，同口径重判 07-22）
| benchmark | no-anchor (P33) | with-anchor (ours) | Δ |
|---|--:|--:|--:|
| HRBench4K | 76.38 | 76.25 | −0.13 |
| HRBench8K | 73.12 | 73.75 | +0.63 |
| Hallu三均 | 53.43 | 54.86 | +1.43 |
anchor 的 acc 净贡献在噪声带内(±2)。**语言漂移(rollout 含 CJK 比例，机制证据)**：P33(no-anchor) step50/70/90 = 14.1/16.4/26.6%；ours(anchor) = 6.2/8.6/21.9% —— anchor 减缓漂移（中段约减半）。多 seed anchor training 进行中（补 acc 证据）。**措辞红线**：不写"anchor 进一步提升/必要"强 claim（acc 打平）；可写"减缓语言漂移"机制 + 定位为"可选精修/继承性组件"。

**anchor 中段 acc 假设已否定（07-22 同口径 step60 对照）**：P33(no-anchor) step60 = **67.22** vs ours(anchor) FC1 step60 = **66.39**（旧压低 65.99）——no-anchor 中段 acc 甚至略高;step90 也是 P33 略高（66.93 vs 66.69）。→ **anchor 在 acc 上全程打平（甚至 no-anchor 略高），价值纯在减少语言漂移**，坐实 5.9 现有叙事，不必改。

### control-image 消融（2B×unfiltered step90，uniform；对比 full-image vs 退化图像）
| control | Acc | 状态 |
|---|--:|---|
| black (ours) | 66.64 | ✅ 主表 |
| no-image (P34a) | 66.49 | ✅ 07-22（逐项 BLINK57.23/MMStar62.93/V*74.87/Math63.10/HR4K75.50/HR8K75.50/Hallu56.31）⚠️ **len4096**（black/其他 ctrl 是 len6144，口径差） |
| Gaussian blur / degrade (P34b) | — | 待 eval（机器停，重提） |
| Gaussian noise (P34c) | — | 待 eval（机器停，重提） |
观察：no-image(66.49) ≈ black(66.64)，差异在噪声带内 → control 具体构造对 acc 不敏感（但 len 口径差待对齐）。degrade/gauss 待补。

**语言漂移 table（进 paper，用户 07-22 要作 anchor 依据）**：rollout 含 CJK 比例（每 step 256 样本）：

| step | P33 (no-anchor) | ours (anchor) |
|---|--:|--:|
| 10 | 7.8% | 7.0% |
| 30 | 12.5% | 11.7% |
| 50 | 14.1% | 6.2% |
| 70 | 16.4% | 8.6% |
| 90 | 26.6% | 21.9% |

定性例子（P33 step90 rollout output，**paper 里禁中文 verbatim，只能英文描述**）：(a) 整段推理漂成中文（math 题解全中文）；(b) 词级 code-switching（英文推导里混入中文字符，如把 "已知this cosine是 4/5" 这种中英混在一句）。**诚实措辞**：写 "anchor substantially reduces language drift (roughly halved mid-training) and improves generation quality, even though the aggregate accuracy is comparable"，**不写** "collapses without anchor"（ours step90 也 21.9%）。用途 = anchor 的生成语言稳定性依据（acc 打平但漂移明显更少）。

## 14. Divergence 方向消融（forward / JSD / reverse KL，新口径，2026-07-23）

配置：Qwen3-VL-2B × virl39k-filtered × contrast-standard × 90 步，`ra_divergence_alpha`={0.0 forward / 0.5 JSD / 1.0 reverse}。**全部新 7-bench 口径**（HR Average + Hallu 三均，去 MMBench/POPE/Zoom），与主表一致——三方逐 benchmark 从各自 eval 目录重算：

| divergence | BLINK | MMStar | V* | MathVista | HR4K | HR8K | Hallu | 7-Acc |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| forward KL (α=0, ours) | 58.18 | 63.87 | 78.01 | 67.00 | 76.38 | 73.12 | 52.93 | **67.07** |
| JSD (α=0.5) | 57.08 | 62.33 | 76.44 | 65.70 | 75.37 | 73.00 | 53.86 | **66.25** |
| reverse KL (α=1) | 56.92 | 61.07 | 76.44 | 64.00 | 73.62 | 70.50 | 50.86 | **64.77** |

**结论**：forward(ours) 67.07 > JSD 66.25 > reverse 64.77。forward KL（mode-covering，target 在前 KL(target‖student)）最佳；reverse KL（mode-seeking，多数蒸馏工作的默认方向）最差；JSD（generalized α=0.5）居中。→ 支持 forward KL 作默认。forward 67.07 ≈ 主表 ours 67.04（standard≈uniform，w_t 打平）。

来源目录：forward=`outputs_api_server/contrast_standard_virl39k_90step_server_eval`；reverse=`outputs_vllm_curated/contrast_reversekl_2b_virl39k_step90`；JSD=`outputs_api_server/jsd_2b_virl39k_90step_step90_eval`。`ra_divergence_alpha` 是 loss 选择器（0/0.5/1 硬切换 forward/JSD/reverse），与 `ra_contrast_alpha`（对比锐化强度）是无关的两个参数。（旧口径同实验为 forward 70.68/JSD 69.74/reverse 68.31，趋势一致；JSD 首评曾被 judge 429 污染成"暴跌"，已重判平反。）

## 15. VDH — 视觉依赖 token 可视化（paper 素材，2026-07-23）

定义：per-token $\Delta_t=\log p(y_t\mid\text{img})-\log p(y_t\mid\text{black})$，越大=该 token 越依赖看真图。对同一段 response（ours 的真实 rollout）用三个 teacher（base/OPSD/ours）各算一遍。脚本：`scripts/visual_dependency_highlight.py`（GPU 算 Δ，批量走 `docs/vdh_manifest.jsonl`）+ `scripts/render_vdh.py`。数据：`docs/vdh_out/*.json`（13 样本，Qwen3-VL-2B，base=vanilla_nothink / opsd=answerhint_qwen35_unfiltered / ours=cons_qwen35_seedA 的对应 checkpoint）。

- **用户选中 case：`MMStar_282`**（2026-07-23）。
- **advantage = ours − max(base,opsd)**（ours 比两个 baseline 都强调更多的 token）排序（visual token 平均）：MathVista_821 +0.56 / 233 +0.38 / MMStar_316 +0.35 / VStar_25 +0.22 / **MMStar_282 +0.20** / VStar_136 +0.13。反例（advantage≤0，靠世界知识非看图）：MathVista_29/745（名人年龄题）、MMStar_309。
- 展示形式历史：① matplotlib token 底色高亮（有 overlap，弃）② 折线 ③ 柱状聚合（visual vs non-visual token μΔ：ours 0.79 vs base 0.62 vs opsd 0.65；非视觉 token 三方≈0.09——ours 特异增强视觉 token；**用户否，要 token-level**）④ **HTML token 高亮**（`docs/vdh_*.html`，`white-space:pre-wrap` 天然不 overlap，用户认可这个方向）⑤ advantage 绿色行（ours−max(base,opsd)）。
- 待定：最终进 paper 的样本 + 转 PDF（LaTeX 吃不了 HTML）。措辞红线：单样本只能 "illustrative/consistent with"，量化证据在柱状聚合。

## 16. Vision-OPD 对比（no-supervision baseline，2026-07-30 整理；源 Vision-OPD-exp/docs/plan_beat_visionopd_20260727.md）

Vision-OPD = **bbox-supervised** OPSD（用标注 crop teacher，属监督信号）；\method = **no-supervision**。对比只在 **4B、两底座**（Qwen3-VL-4B / Qwen3.5-4B）有数据。**Vision-OPD 行是我们的 reproduction，从未有官方 ckpt**（Qwen3-VL-4B = 官方代码训 yijiangli step65；Qwen3.5-4B = 我们训 trial301761390 step62）。

### 16a. VLMEvalKit 7-bench（\method uniform s90 × virl39k；temp0/pp1.5/4096，gpt-5.4-mini judge）
| 底座 | base | \method | Vision-OPD (repro) | \method − Vision-OPD |
|---|--:|--:|--:|--:|
| Qwen3-VL-4B | 71.30 | 72.75 | 70.27 | **+2.48** |
| Qwen3.5-4B | 73.94 | 75.89 | 73.06 | **+2.83** |
- 净提升（vs base）：\method Qwen3-VL-4B +1.86 / Qwen3.5-4B +0.33；Vision-OPD-repro −1.03 / −3.39（**7-bench 全线低于 base**）。

### 16b. Native 定位套件（V*/HR4K/HR8K/ZoomBench；同管线 v3-fixed judge，temp0 seed42）
| 底座 | 方法 | V* | HR4K | HR8K | Zoom |
|---|---|--:|--:|--:|--:|
| Qwen3-VL-4B | \method | 85.86 | 80.50 | 77.25 | 41.42 |
| Qwen3-VL-4B | Vision-OPD (repro) | 84.82 | 81.00 | 77.12 | 53.25 |
| Qwen3.5-4B | \method | 85.34 | 86.50 | 81.88 | 52.19 |
| Qwen3.5-4B | Vision-OPD (repro) | 90.58 | 82.12 | 79.62 | 59.05 |
- \method − Vision-OPD：Qwen3-VL-4B → V* +1.04 / HR4K −0.50 / HR8K +0.13 / **Zoom −11.83**；Qwen3.5-4B → V* −5.24 / HR4K +4.38 / HR8K +2.26 / **Zoom −6.86**。

### 16c. 叙事（paper section 用）
1. \method（无监督）7-bench 上高出 Vision-OPD reproduction **+2.9 / +3.7**。
2. **"拿通用换定位"**：Vision-OPD 的 7-bench 全线低于 base（用通用能力换定位）；\method 通用小幅正向、定位大体持平。
3. \method 唯一实质短板 = **ZoomBench**（−6.9 / −11.8），来自 Vision-OPD 的 **bbox 监督 crop 机制**在定位任务上的优势。
4. paper 定位：\method 作为 **no-supervision baseline**，不引入任何额外标注，即可保持/提升通用能力并在多数定位维度持平。

### 16d. 措辞红线（写 tex 必守）
- Vision-OPD 行一律写 "**our reproduction of Vision-OPD**"，不得暗示是官方 ckpt/官方数字。
- **两个口径（VLMEvalKit 7-bench 表 16a vs native 定位表 16b）分开呈现、绝不混排**（口径不同：judge/温度/采样均不同）。
- native 口径已与 Vision-OPD 官方 report 对齐（Qwen3-VL-4B 四项 ±0.5 内、Qwen3.5-4B −1.6 内），可信。
- bbox = 标注监督，\method **不引入**（no-supervision 是核心 claim）。不写"beats/solves"类禁用词。

### 16e. 综合表（用户 2026-07-30 要求：把 16a+16b 合并成一张，格式同主表 Table 1）
列 = Model / Method / BLINK / MMStar / V* / MathVista / HR4K / HR8K / HalluB / Acc / ZoomBench。**这张替换原来的两张（Table 5 tab:vision-opd-seven-benchmark + Table 6 tab:vision-opd-localization）。**

| Model | Method | BLINK | MMStar | V* | MathVista | HR4K | HR8K | HalluB | Acc | Zoom |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Qwen3-VL-4B | base | 67.18 | 68.93 | 80.63 | 73.90 | 79.88 | 73.75 | 54.81 | 71.30 | 44.14 |
| Qwen3-VL-4B | \method | 66.12 | 70.53 | 84.29 | 77.20 | 80.88 | 70.37 | 59.87 | **72.75** | TBD |  ← 08-06 换成 vopd6k 训（旧行 66.86/…/73.16/43.08 是 virl39k 训，与本表 Vision-OPD 数据不符，已作废）
| Qwen3-VL-4B | Vision-OPD (repro) | 63.76 | 65.67 | 83.25 | 71.40 | 76.50 | 75.25 | 56.03 | 70.27 | 52.90 |
| Qwen3.5-4B | base | 64.97 | 73.00 | 82.20 | 81.70 | 81.38 | 74.00 | 60.36 | 73.94 | 52.43 |
| Qwen3.5-4B | \method | 67.65 | 74.73 | 84.29 | 82.70 | 85.88 | 73.62 | 62.35 | **75.89** | TBD |  ← 08-06 换成 vopd6k 训（旧行 …76.78/53.49 是 virl39k 训，已作废）
| Qwen3.5-4B | Vision-OPD (repro) | 62.65 | 70.93 | 86.91 | 79.20 | 82.38 | 72.00 | 57.35 | 73.06 | 59.05 |

口径说明（进 caption/脚注）：
- **BLINK…HalluB + Acc = VLMEvalKit seven-benchmark**（temp0/pp1.5/4096，gpt-5.4-mini judge）；Acc = 这 7 项均值。
- **ZoomBench = native 口径**（v3 judge，temp0 seed42）；VLMEvalKit 不含 Zoom，故单列并用脚注注明口径不同。
- ⚠️ 此表的 **V*/HR4K/HR8K 是 VLMEvalKit 口径**（与 §16b native 表的同名列数值不同，如 ours-VL-4B 这里 V\*=83.77 vs native 85.86）——合并表统一走 VLMEvalKit，别把 §16b 的 native V*/HR4K/HR8K 混进来。
- Vision-OPD 行仍是 our reproduction（措辞红线同 §16d）。

## 17. Eval 可复现性验证(2026-08-04/05,devbox 提交 ruby 665 1-GPU mlx)

**目的**:用户要求复刻主表 ours(2B) 67.04 那行,验证 eval 管线可复现;后扩展到 OPSD 与 4B。
**协议**:同 ckpt、同 7-bench、BACKEND=vllm_server、**系统栈 vllm 0.11**(不激活 conda,与原 07-20 serve 日志栈一致)、seed 未动(vllm serve 默认 seed=0,两边一致)、新 tag `*_rerun0804` 不覆盖原结果。

| 复刻 | ckpt | 原 Acc | rerun Acc | Δ | V* Δ |
|---|---|--:|--:|--:|--:|
| 2B ours (FC1 s90) | `...contrast-uniform-...UNFILTERED1img-150step-keepall-trial301783374/global_step_90` | 67.04 | 66.62 | −0.42 | **−3.14**(78.01→74.87) |
| 2B OPSD (FC4 s90) | `...baseline-...150step-keepall-trial301783374/global_step_90` | 64.18 | 64.37 | +0.18 | +1.57(75.39→76.96) |
| 4B ours (uniformweight s90) | `...contrast-standard-uniformweight-Qwen3-VL-4B-...90step-trial301829143/global_step_90` | 73.16 | 73.26 | +0.10 | −0.52(83.77→83.25) |

**结论**:
1. 管线可复现:21 项里 20 项 ±1.6 内(4B 几乎逐字,HR4K Δ=0.00)。整体 Acc ±0.42 内。
2. **V\* 波动是 2B 特有的模型边缘性**:2B 两次一负一正(−3.14/+1.57),4B 稳(−0.52)。2B 逐样本 diff:191 样本 58 条文本不同、仅 8 条判定翻转(7 对→错 1 错→对),全是颜色/方位感知题、预测开头相同中途分叉 = temp0 下 vllm 并发 batching 数值非确定性。FC1 曲线自身 V\* 波动带 71.73~78.53 亦印证。
3. 排除项:seed(两边默认 0)、采样 config(逐字一致)、eval 代码(vlmeval 核心 07-21 后零改动;08-03 的 eval_via_vllm_server 改动仅 video 参数、空默认不进 image 路径)。
4. ⚠️ 主表 OPSD 行(64.89)= S2b seed1234 ckpt **已被 prune,无法复刻该行本身**;FC4 是同配方默认 seed 的现存 ckpt。
5. rerun 结果快照:各 eval 目录下 `normal_scoring*snapshot*`;paper 数字不动(原值仍有效,rerun 证明其稳健)。
