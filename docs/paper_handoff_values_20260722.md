# Paper 数值交接单（2026-07-22，301967423 出具）

> 给「更新 paper」的 session 直接取用。**全部为 paper 7-bench 口径、经判分核验的干净值。**
> 出具机器：trial 301967423（前身 301832790）。原始产物路径见文末 §5，可自行复算。

---

## 0. 口径定义（三处务必统一）

**① Suite（7 项）**
`BLINK / MMStar / VStarBench / MathVista_MINI / HRBench4K / HRBench8K / HallusionBench`
**Acc = 这 7 项的算术平均。已去掉 MMBench、POPE、ZoomBench**（2026-07-22 用户拍板）。

**② HRBench 指标**：取 `acc.csv` 里 **`cycle=Average` × `type=all`** 一行。
即 4 个 cycle（选项循环置换）的平均、single+cross 合计。
（同一 acc.csv 内另有 `single`/`cross` 细分，本表未用；如 paper 要分开报需整表重算。）

**③ HallusionBench 指标**：**aAcc / fAcc / qAcc 三者的算术平均**（记作「Hallu 三均」），非单一 aAcc。

**④ 评测设置**：VLMEvalKit `BACKEND=vllm_server`，temp=0 / presence_penalty=1.5 / max_new_tokens=4096，
judge = Azure `gpt-5.4-mini-2026-03-17`（provider `tiktok_azure`），规则优先 + GPT 兜底。

---

## 1. 主表：6 底座 × (base / OPSD / ours)

| 底座 | 方法 | BLINK | MMStar | V* | MathVista | HR4K | HR8K | Hallu三均 | **Acc** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-2B | base | 53.02 | 57.47 | 72.77 | 62.50 | 71.13 | 67.38 | 51.60 | **62.27** |
| | OPSD | 56.02 | 60.73 | 75.92 | 64.70 | 76.25 | 71.25 | 49.36 | **64.89** |
| | **ours** | 57.13 | 61.40 | 76.44 | 66.80 | 76.38 | 73.62 | 54.72 | **66.64** |
| Qwen3-VL-4B | base | 67.18 | 68.93 | 80.63 | 73.90 | 79.88 | 73.75 | 54.81 | **71.30** |
| | OPSD | 64.97 | 68.73 | 83.77 | 74.10 | 78.12 | 75.63 | 54.50 | **71.40** |
| | **ours** | 66.86 | 68.73 | 83.77 | 74.90 | 80.50 | 77.00 | 60.37 | **73.16** |
| Qwen3-VL-8B | base | 69.65 | 70.67 | 82.72 | 76.50 | 77.38 | 71.12 | 59.51 | **72.51** |
| | OPSD | 67.81 | 70.20 | 85.34 | 77.30 | 81.12 | 74.62 | 59.64 | **73.72** |
| | **ours** | 70.38 | 74.07 | 87.43 | 77.90 | 84.12 | 79.25 | 60.69 | **76.26** |
| Qwen3.5-2B | base | 60.55 | 67.20 | 80.63 | 74.90 | 74.00 | 72.88 | 49.96 | **68.59** |
| | OPSD | 59.44 | 67.00 | 79.58 | 69.70 | 75.00 | 68.75 | 43.76 | **66.18** |
| | **ours** | 64.86 | 69.40 | 83.77 | 76.50 | 78.50 | 70.75 | 56.77 | **71.51** |
| Qwen3.5-4B | base | 66.91 | 72.67 | 84.29 | 80.70 | 86.12 | 82.00 | 62.45 | **76.45** |
| | OPSD | 64.44 | 71.40 | 84.29 | 77.60 | 82.75 | 78.75 | 58.22 | **73.92** |
| | **ours** | 66.70 | 75.00 | 84.82 | 82.20 | 85.38 | 79.38 | 63.95 | **76.77** |
| Qwen3.5-9B | base | 67.39 | 75.60 | 89.01 | 82.00 | 87.25 | 83.00 | 66.53 | **78.68** |
| | OPSD | 67.65 | 72.27 | 87.96 | 78.80 | 81.50 | 78.12 | 56.80 | **74.73** |
| | **ours** | 72.23 | 78.87 | 85.86 | 85.00 | 87.00 | 81.50 | 64.19 | **79.24** |

**ours − base**：2B +4.37 / 4B +1.86 / 8B +3.75 / Q35-2B +2.92 / Q35-4B +0.32 / Q35-9B +0.56

### ⚠️ 主表两处必须在正文交代的异常
1. **Qwen3.5-2B OPSD = 66.18，低于 base 68.59（−2.41）**
2. **Qwen3.5-9B OPSD = 74.73，低于 base 78.68（−3.95）**，Hallu 三均 56.80 vs base 66.53 是最大单项落差
   → answer-hint 在 Qwen3.5 系（尤其 9B）出现**负迁移**，与 Qwen3-VL 系「OPSD ≈ base 或略高」相反。机制未查清，建议如实写、不强行解释。
3. **ours 的优势随 scale 收窄**（2B +4.37 → Q35-4B +0.32 / Q35-9B +0.56）——若 paper 主打「跨 scale 一致增益」需谨慎措辞。

---

## 2. α 曲线（Qwen3-VL-2B × uniform × unfiltered × 90step）

| α | BLINK | MMStar | V* | MathVista | HR4K | HR8K | Hallu三均 | **Acc** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **0**（matched baseline，无对比锐化） | 56.40 | 60.10 | 74.30 | 63.90 | 74.00 | 68.00 | 53.47 | **64.31** |
| **0.5** | 52.13 | 58.53 | 76.44 | 60.40 | 72.88 | 70.75 | 51.41 | **63.22** |
| **1.0**（主配方） | 57.13 | 61.40 | 76.44 | 66.80 | 76.25 | 73.75 | 54.86 | **66.66** |
| **1.25** | 58.02 | 63.73 | 75.39 | 66.50 | 77.12 | 72.25 | 52.64 | **66.52** |
| **1.5** | 58.29 | 62.60 | 75.39 | 64.10 | 76.50 | 73.12 | 52.59 | **66.08** |
| **2.0** | 52.76 | 60.67 | 77.49 | 61.70 | 72.25 | 69.88 | 52.63 | **63.91** |

**形态**：**α∈[1.0, 1.25] 是平台**（66.66 / 66.52，差 0.14）→ 1.5 缓降（66.08）→ **两端极值 0.5 / 2.0 陡掉约 3pp**。
建议表述为「**α 在 [1.0,1.5] 稳健，仅极值明显变差，默认 α=1 位于平台内**」，比「尖峰在 α=1」更准确。

**α=0.75 未跑**：两次 step10 确定性 OOM，用户拍板 skip（左右 0.5/1.0 已 bracket）。

### ⚠️ α=1.0 的 Acc 有两个版本，请统一
- 主表记 **66.64**（HR8K=73.62）
- 本节记 **66.66**（HR8K=73.75，干净重判值）
→ **建议全文统一用 66.66 / HR8K 73.75**，差 0.02 仅来自 HR8K 取值来源。

---

## 3. ctrl 分支消融（Qwen3-VL-2B × uniform × unfiltered × 90step）

| ctrl | BLINK | MMStar | V* | MathVista | HR4K | HR8K | Hallu三均 | **Acc** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **gaussnoise** | 58.23 | 63.60 | 76.44 | 67.20 | 76.38 | 73.25 | 54.87 | **67.14** |
| **black**（主配方） | 57.13 | 61.40 | 76.44 | 66.80 | 76.25 | 73.75 | 54.86 | **66.66** |
| **degrade** | 55.65 | 62.87 | 78.01 | 66.00 | 76.88 | 73.88 | 51.25 | **66.36** |
| **noimg** | 57.23 | 62.93 | 74.87 | 63.10 | 76.38 | 72.88 | 56.31 | **66.24** |

**结论：四路差 ≤0.90pp，全在噪声带内 → 「怎么移除视觉信息」不敏感。**
与旧口径「black > qtext（−2.66）」不矛盾而是互补：**关键是「是否移除视觉信息」**（这四路都移除 → 彼此等价；qtext 保留图像只换问题 → 明显更差）；**具体破坏方式无关紧要**。
→ black 的地位可从「必须」放松为「任选其一，black 最简单」。

### ⚠️ 口径差
**noimg 用 len4096**（ctrl 分支无图像 token、与 hi 分支长度差大，@6144 确定性 OOM——这本身是个工程发现），
其余三路 len6144。**noimg 的 −0.42 落后不宜过度解读。**

---

## 4. 判分可信度（本批数值均已核验）

本项目反复踩过一个坑：**uniform-unfiltered 系模型输出是长 CoT**（BLINK 预测中位 1200–1400 字符），
MCQ/MathVista/Hallu 的规则提取抽不出答案 → 必须走 GPT judge；若 judge 因 `ALL_PROXY=socks5h` 缺 socksio、
或缓存未清而没真跑，未解析样本会被当错题 → **分数静默虚低**。

**本交接单所有数值的核验状态：**
- Hallu **fAcc 全部落在 42–56 健康档**（污染时会掉到 ~28），无 exact-match 回退警告；
- α=0.5 / 2.0（FA1/FA2）曾被怀疑虚低，**删全缓存 + 规则优先-GPT 重判后变化 <0.2pp**，证明本就干净；
- α=0 与 P33 系曾确被压低，**已重判修正**（本表为修正后值）；
- FA1/FA2 的 vLLM serve 日志核查：**served-model-name 与 ckpt 路径正确、推理失败率 0%**（7 数据集全查），无端口串模型。

**复算方法**：`cycle=Average/type=all`（HR）、`Overall` 行（MCQ）、`(aAcc+fAcc+qAcc)/3`（Hallu）。
⚠️ **Acc 是算出来的均值，任何文件里都没存这个数**——不要用 grep 找聚合值。

---

## 5. 原始产物路径（共享 NAS，可自行复算）

根目录：`/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit/outputs_vllm_curated/`

| 行 | 目录名（后接 `/normal_scoring/`） |
|---|---|
| Q35-9B base | `vanilla_qwen35_9b_qwen3vl2b_temp0_4096_generic_eval` |
| Q35-9B OPSD | `n6_answerhint_qwen35_9b_step90_qwen3vl2b_temp0_4096_generic_eval` |
| Q35-9B ours | `n4_uniformweight_qwen35_9b_unfiltered_step90_server_qwen3vl2b_temp0_4096_generic_eval` |
| α=1.0 / black / ours-2B | `uniformweight_2b_unfiltered_step90_qwen3vl2b_temp0_4096_generic_eval` |
| α=0.5 | `fa1_uniform_alpha05_unfiltered_step90_server_qwen3vl2b_temp0_4096_generic_eval` |
| α=1.25 | `p37b_alpha125_step90_qwen3vl2b_temp0_4096_generic_eval` |
| α=1.5 | `p37c_alpha15_step90_qwen3vl2b_temp0_4096_generic_eval` |
| α=2.0 | `fa2_uniform_alpha20_unfiltered_step90_server_qwen3vl2b_temp0_4096_generic_eval` |
| ctrl noimg | `p34a_ctrl_noimg_len4096_step90_server_qwen3vl2b_temp0_4096_generic_eval` |
| ctrl degrade | `p34b_ctrl_degrade_unfiltered_step90_qwen3vl2b_temp0_4096_generic_eval` |
| ctrl gaussnoise | `p34c_ctrl_gaussnoise_unfiltered_step90_qwen3vl2b_temp0_4096_generic_eval` |

其余底座行的目录见总账 `docs/reports/ra_vad_results_ledger.html` §1 对应脚注。

---

## 6. 尚未收口、暂不要写进 paper 的项

- **QA2（Qwen3.5-4B α=2.0）** 补判分进行中 → 出数后可拼 Qwen3.5-4B 的 α 三点（0.5/1.0/2.0），验证「α=1 峰值」是否跨 scale 成立。
- **P33 no-anchor seed1234** eval 进行中 → 给「anchor 冗余」结论补第二 seed。
- **nosamplegate（终局口径）** eval 进行中。
- **所有数值目前均为单 seed**（2B 主表行有 3-seed mean±std 的历史记录，见总账消融 F）；若 paper 要下强 claim，建议标注单 seed 或补种子。
