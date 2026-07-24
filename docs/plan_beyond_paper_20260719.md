# Beyond-paper 计划：OPSD×GRPO 合体 + 双 ctrl（2026-07-19 立项）

> 用户两问：① 把现在的 opsd(ours) 和 GRPO 结合，会不会比 GRPO 更好？② qtext 和 black 两路 ctrl
> 能不能合出更好的结果？本文档记录假设、实验清单、判定标准与进度。
> **口径约定**：所有新实验以终局配置为基（uniform × unfiltered × 2B × β=0.1 × EOS豁免 × 90步，
> len6144，8 卡），eval 用主口径（temp0/4096），与主表可直接比。噪声带 ±2pp（2B composite seed ±3pp，
> 关键结论需双 seed）。

---

## 方向一：ours × GRPO 合体（G 系）

### 已有证据（为什么可能有互补）
- 算力：ours 90 步 ≈ GRPO 1ep(464步) 的 1/5，7-bench 只差 ~1pp（70.04-70.68 vs 71.66）。
- 收益面互补：GRPO 强在 MMStar/MMBench/HR 系；ours 强在 BLINK/Hallu/Zoom（GRPO×virl39k Zoom 38.34 是全表最低，ours 41.9-43.6）。
- 机制正交：GRPO 需要 verifiable reward、学"答对"；ours 无 reward、学"看图"。
- ⚠️ 定位红线（compare 文档 682 行已研判）：按信号门控 KD/PG 与 **AOPD (arXiv 2605.06387)** 高度相似
  ——差异化必须讲清：AOPD 门控轴=competence gap（哪里能力不足），我们=modality dependence（哪里需要看图）。
  借鉴其工程结论（forward KL、top-K）不算冲突。

### 实验清单（按优先级）

| # | 实验 | 配方 | 成本 | 判定 | 状态 |
|---|---|---|---|---|---|
| **G1** | **序贯：ours→GRPO**（ours-uniform-unfiltered step90 作 init，跑 GRPO×virl39k 1ep） | `run_experiment_grpo_baseline.sh` + `MODEL_PATH=P26/global_step_90`，零代码 | 8卡 ~5h | vs GRPO-from-base 1ep（71.66）：composite ≥+1pp 或同分但 Zoom 回血（>41）即"合体有益"；若 ≤71.66 且 Zoom 也掉 → 序贯无益，转 G3 | 🏃 301832790 07-19 22:3x 启动 |
| G2 | 反向序贯：GRPO→ours（GRPO 1ep ckpt 作 init 跑 uniform 蒸馏 90 步） | 同样零代码，`MODEL_PATH=GRPO step464 merged` | 8卡 ~2h | 与 G1 对照：谁先谁后重要吗；若 G2>G1，说明蒸馏适合做"后处理打磨" | ⏳ G1 出数后跑 |
| G3 | 联合 loss：`L = pg + λ·kd`（每步同时算 GRPO 目标与蒸馏目标，λ∈{0.3,1.0}） | 需改 dp_actor（vopd 模式现在是替换 pg，需加混合模式）；~1 天代码+冒烟 | 8卡 ~6h/λ | 超过 max(G1,G2) 才值得；否则序贯就够且更简单 | 💤 等 G1/G2 判定 |
| G4 | 门控路由：ra_margin 高的样本走 KD、低的走 PG（AOPD 结构，modality 轴差异化） | 代码量同 G3；novelty 风险最高，写作负担大 | — | 只有 G3 显示混合>序贯、且效应来自互补而非平均时才立项 | 💤 远期 |

**G1 细则**：EXPERIMENT_NAME=`Vision-OPD-grpo-from-ours-uniform-Qwen3-VL-2B-virl39k-UNFILTERED1img-1ep-trial301783374`；
数据同 unfiltered parquet（GRPO baseline 此前用 filtered 464 步——⚠️ 对照口径：G1 用 unfiltered 则对照行应为
"GRPO-from-base×unfiltered"，**若无此行需补跑 G1b 对照（GRPO from base，unfiltered，1ep）**，否则 filtered
的 71.66 只能作参考线不能作严格对照）。reward=mcq_exact（virl39k 是 MCQ，现成）。

### 决定树
G1 赢 → 补 G1b 严格对照 + 双 seed → 若稳，即得"1/5 蒸馏预算 + 1ep GRPO > 纯 GRPO"的 practical recipe（可作后续 paper 核心）。
G1 平/输 → G2；两者都无益 → 序贯路线关闭，G3 联合 loss 是唯一出路（成本高，先小 λ 扫描）。

### G1 结果（2026-07-20/21，301832790）—— ✅ 9-bench 全出，判定完成
- **训练完成**：194/194 步（1ep，8:39:49），末步 critic/score/mean=0.523，已 merge global_step_194。
- **完整 9-bench（temp0/4096，GPT judge）**：

| | BLINK | MMStar | MMBench | V* | MathVista | HR4K | HR8K | POPE | Hallu(a/f/q三均) | **Zoom** |
|---|---|---|---|---|---|---|---|---|---|---|
| **G1**（ours→GRPO 序贯） | 57.23 | 62.33 | 75.69 | 78.01 | 67.40 | 78.50 | 75.62 | 87.71 | 55.59 | **44.97** |

- **旧口径 7-bench（含 MMBench）= 70.68** —— 与主表 ours-uniform(70.68) **完全持平**。
- 新口径 7-bench（Hallu 三均替 MMBench、去 POPE）= 67.81。
- **ZoomBench 44.97** ✅：纯 GRPO×virl39k 38.34（全表最低）、ours-uniform 41.9–43.6 → 序贯后 **全表最高，+6.6pp vs 纯GRPO**。
- **✅ 判定（按决定树）**：composite 70.68 ≤ GRPO-from-base 71.66（filtered 参考线，−0.98pp 噪声带内，**未净增**）
  但 **Zoom 回血支线明确成立（44.97 ≫ 41）** → 落"合体有益（Zoom 维）"。**composite 未提升、收益集中在 Zoom**
  是关键限定——不是"1/5 预算 + 1ep > 纯 GRPO"的全面胜利，而是"序贯合体把 ours 的 Zoom 长处叠加到 GRPO 上、
  且不掉 composite"。
- **⚠️ 口径限定**：71.66 是 **filtered** GRPO-from-base，G1 是 **unfiltered** → composite 对照不严格。
  **下一步 = G1b（GRPO-from-base × unfiltered 1ep 严格对照）**：只有 G1b 出来才能确认 G1 的 composite 持平/Zoom 回血
  是"合体贡献"而非"unfiltered 数据本身的效应"。G1b 是 8 卡 GRPO ~5h，排在本机 α=0→P33 链之后（或交空闲姊妹机）。
- **决定树落点**：G1 赢（Zoom 支线）→ 待 G1b 严格对照 + 双 seed 定稿；若 G1b 也显示 Zoom 回血来自合体而非数据，
  即得"序贯蒸馏打磨 GRPO 的 Zoom 短板"的 practical recipe。

---

## 方向二：black × qtext 双 ctrl（Q 系）

### 已有证据
- 四象限（2026-07-16）：black 70.65 / qtext 67.99，错误互补——仅black对 657 vs 仅qtext对 494，
  oracle union 77.90（+6.8pp 上界；含噪声红利，真实可得 1/3-1/2）。
- 机制：black ctrl=「图像增益」信号，qtext ctrl=「问题增益」信号——正交轴。
- qtext 弱在纯视觉感知（Forensic/Visual_Similarity 贡献 BLINK 净差 72%）；MathVerse split 假设
  （qtext 在 Text-Dominant 更稳）**尚未验证**（E 组 qtext×MathVerse 一直没跑成，Q0 收编）。

### 实验清单

| # | 实验 | 配方 | 成本 | 判定 | 状态 |
|---|---|---|---|---|---|
| Q0 | 补证据：qtext×MathVerse split + 现成 black/qtext ckpt 的真实 ensemble 探针（BLINK 子集 logit 平均，离线 HF 前向） | 1 GPU 半天；ensemble 探针给出"真实可得增益"（oracle 上界的现实折扣率） | 低 | ensemble 真实增益 <1pp → 合体上界太小，Q 系降级；≥2pp → Q2 立项依据 | ⏳ 本机 G1 期间可跑（1 卡） |
| **Q1** | **ctrl 课程（零代码）**：uniform×unfiltered 90 步，前 45 步 ctrl=black、后 45 步 resume 改 ctrl=qtext（Q1a）；反序 Q1b | resume 时换 `ra_ctrl_mode` CLI 即可（config 每次启动重读）；零代码 | 8卡 ~2h/个 | vs 纯 black（70.04）：≥+1.5pp 才算信号；两序都平 → 信号不在时间维，直接 Q2 | ⏳ G1 后排 |
| Q2 | **双 tilt target（核心方法）**：`target ∝ softmax(lp_hi + α₁(lp_hi−lp_black) + α₂(lp_hi−lp_qtext))`，α₁=α₂=0.5 起步 | 需改 ra_vad/dp_actor：每步多一次 qtext-ctrl teacher 前向（+~30% 步时）；含 β mask 作用于合成 tilt | 代码 ~1 天 + 8卡 ~3h/run | vs black 单 ctrl：composite ≥+1.5pp 且 BLINK/MathVerse-TextDom 双向不掉 → 双 ctrl 成立 | 💤 等 Q0/Q1 |
| Q3(远期) | 逐 token ctrl 路由：按 |lp_hi−lp_black| vs |lp_hi−lp_qtext| 逐 token 选更大的 tilt | 代码同 Q2 基础上小改 | — | Q2 成立且分析显示两信号在不同 token 段各占优时才做 | 💤 |

### 决定树
Q0 的 ensemble 真实增益是闸门：小 → 整个方向降级为 paper 的 future work 一句话；大 → Q1 快筛时间维，
Q2 是正式方法（若成，即"multi-ctrl contrastive distillation"，可作后续 paper 的第二核心）。

---

## 附：α=0 matched baseline + P33 no-anchor 结果（2026-07-21，主 paper 消融，非 beyond-paper，但记此备查）

> ⚠️ 这两个 eval 首跑撞 judge-API→exact-match 污染（**VStar/HRBench 也走 judge，只 POPE 纯规则**），
> 已全部删缓存重判，下表是**从 judge log RESULT_JSON 提取的干净值**（勿读 `*_acc.csv`，可能是旧污染值）。

| bench | base | **α=0** | **P33(no-anchor)** | ours(α=1) |
|---|---|---|---|---|
| BLINK | 53.02 | 55.29 | 58.71 | 59.92 |
| MMStar | 57.47 | 59.33 | 62.93 | 64.07 |
| MMBench | 78.09 | 76.03 | 78.09 | 79.81 |
| VStar | 72.77 | 74.35 | 75.92 | 76.44 |
| MathVista | 62.50 | 64.50 | 67.0 | 67.30 |
| HR4K | 71.13 | **62.62** | 72.0 | 77.25 |
| HR8K | 67.38 | **57.38** | 67.75 | 72.25 |
| Hallu三均 | 51.60 | **29.35** | 38.97 | 54.39 |
| **7-bench均** | 66.05 | **64.21** | 68.91 | **70.68** |
| Zoom | 42.49 | 39.17 | 43.31 | 43.55 |

**结论**：① **α=0（去对比锐化、纯自蒸馏）= 64.21 < base 66.05**——纯 T=2软化+forward-KL 是"模糊化"，
无对比锐化补偿则**主动伤害**（HR/Hallu 暴跌，预测经核验连贯、非模型损坏，是感知一致性真实退化）。
**ours−α=0 = +6.47pp = 对比锐化的真实净贡献，且承重**（matched baseline 完美隔离，排除 reweight/自蒸馏功劳）。
② **P33（去 anchor，纯 softmax(lp_hi−lp_ctrl)）= 68.91**，介于两者间，去 anchor 仅 −1.77——纯对比比值已回收
大部分收益，anchor 是温和精修。**待回填主表/ledger 消融区（α 曲线 + guarded-tilting 邻近）+ paper_notes §3**。

## 资源与推进
- 本机（301832790，8卡）：今晚 G1；明天按决定树推进（G1b/G2/Q1 各 ~2-5h）。Q0 用 G1 期间的空隙 1 卡。
- 代码类（G3/Q2）：立项后先出 design note 给 Codex 审一遍再动 verl。
- 进度与结果全部回写本文档 + queue.md 登记（防撞车规则不变）。
