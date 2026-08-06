# 交接：Vision-OPD6K 对照表的 **native 口径** 版本（给 codex 建表用）

用户 2026-08-06 指令：为 paper 现有的 “Generalization to Vision-OPD6K and comparison with Vision-OPD”
表（VLMEvalKit 7-bench 口径，`main_arxiv_v1.tex` L818~）**新建一张平行表，记录同一批模型在 native 口径下的结果**。
本文件由 devbox 汇总数字与缺口；**codex 负责建表**（tex + 本文档回填）。

---

## 0. 两种口径的区别（写 caption 时必须讲清）

| | VLMEvalKit 7-bench（现有表） | **native（本表）** |
|---|---|---|
| 推理 | VLMEvalKit `run.py` + vLLM server，temp0/max4096，presence_penalty 1.5 | `eval/infer.py` 直连 vLLM，temp0/seed42/max8192 |
| prompt | VLMEvalKit 模板（“Question: … Please select…”） | VisionOPD 官方模板（CoT 风格，`vstar.json`/`hr_bench_*.json`/**`zoombench.json`**） |
| 判分 | 规则 fast-path + `gpt-5.4-mini` judge | `judge_qwenlm.py` **v3-fixed**（三处 bug 已修）+ `cal_acc.py` |
| 覆盖 | 7 个 benchmark | **V\* / HR4K / HR8K / ZoomBench 四项**（native 无 BLINK/MathVista/HalluB） |

⚠️ **两口径同一模型可差 5pp 以上**（prompt 风格 CoT vs 直答，历史已记：HRBench ~5pp gap）。
**两表之间任何单元格都不可相减**；native 表只在其内部行间比较。
⚠️ ZoomBench 必须用 `zoombench.json`（CoT 版）；`zoombench_mcq.json` 低约 7pp，永不混用。

---

## 1. 数字（devbox 2026-08-06 实测汇总，`cal_acc.py` 现算）

### Qwen3-VL-4B

| 行 | V\* | HR4K | HR8K | ZoomBench | 数据来源 |
|---|---|---|---|---|---|
| Base | ⚠️ 37.17 | ⚠️ 25.00 | ⚠️ 25.00 | 45.68 | `judge/{vstar,hrbench-*}/Qwen3-VL-4B-base`（⚠️**疑似坏值，勿用**，见 §2）；Zoom=maliva `z4_base_q3vl4b` |
| Vision-OPD (repro) | **84.29** | **81.12** | 76.62 | **53.25** | `e2_visionopd_yjl4b_step65` / maliva `z6_visionopd_yjl4b` |
| ours（**ViRL39K** 训，旧行/仅参考） | 83.77 | 79.88 | **77.38** | 41.42 | `e1_ours_q3vl4b_step90` / maliva `z5_ours_q3vl4b` |
| **ours（Vision-OPD6K 训，本表应采用）** | ⏳ | ⏳ | ⏳ | ⏳ | native job `0eb62f12ece495e1`（V\*/HR4K/HR8K）+ Zoom job `5aa935d7bdb02561`，**在跑** |

### Qwen3.5-4B

| 行 | V\* | HR4K | HR8K | ZoomBench | 数据来源 |
|---|---|---|---|---|---|
| Base | **89.01** | **87.88** | **80.88** | 52.07 | `e5_base_q354b` / maliva `z1_base_q354b` |
| Vision-OPD (repro) | **90.05** | 81.38 | 80.12 | **59.05** | `e4_visionopd_q35repro_step62` / maliva `z3_visionopd_q35repro` |
| ours（**ViRL39K** 训，旧行/仅参考） | 84.29 | 86.50 | 81.75 | 52.19 | `e3_ours_q354b_step90` / maliva `z2_ours_q354b` |
| **ours（Vision-OPD6K 训，本表应采用）** | **84.82** | **85.12** | **80.38** | ⏳ | `vopd6k_q354b_s64_native`（已出）+ Zoom job `4bbc243a89661420`（在跑） |

参考行（非本表必需，供旁证）：`vopd6k_topk10_bs96_s62` = V\* 83.25 / HR4K 82.38 / HR8K 77.62 / Zoom 43.08。

---

## 2. 已知问题与必须遵守的事项

1. **Qwen3-VL-4B base 的 native V\*/HR4K/HR8K 是坏值**：V\* 37.17、HR 双 25.00，而同模型 VLMEvalKit V\* 是 80.63。
   judge 文件 `Qwen3-VL-4B-base_answer.jsonl` 有 **2675 行 / 191 题**（约 14 倍，疑为多次运行累加）。
   → **建表时该三格留空或标 “pending re-run”，绝不能填这三个数**。已知缺口，待补一次干净 native eval。
2. **ours 行必须用 Vision-OPD6K 训的那版**（与同表 Vision-OPD 同数据）。ViRL39K 版仅作对照参考，
   若要并列展示需单独一行并注明数据来源——这正是 7-bench 表 08-06 刚修过的问题（commit `fc9d6ac`）。
3. **Zoom 有两处口径**：paper 现表用的是 ruby 侧历史值（Q3-VL-4B base 44.14 / VOPD 52.90 / ours 43.08；
   Q3.5 base 52.43 / VOPD 59.05 / ours 53.49）；上表 Zoom 列用的是 **maliva 六格同批复测**
   （45.68 / 53.25 / 41.42；52.07 / 59.05 / 52.19）。**同一列必须整列取同一批**，不可混。建议 native 表
   整列采用 maliva 六格批次（同机同期，内部可比）。
4. 数字全部 **n=1 单读**；按 §6 纪律，若某个结论要下判断需 n≥2。

---

## 3. 现在可以确定的结论（供写正文）

- **Vision-OPD 在 native 口径下的 Zoom 优势更明显**：Q3-VL-4B 53.25 vs ours(ViRL39K) 41.42；
  Q3.5-4B 59.05 vs 52.19——与 7-bench 表的结论一致（Zoom 是 bbox-crop 监督的强项，我方已知短板）。
- **Q3.5-4B 上 native 与 VLMEvalKit 结论不完全一致**：native 下 base 的 V\*/HR4K 反而最高
  （89.01 / 87.88），而 VLMEvalKit 下 ours 领先——**这正是需要单独立表、并说明口径差的原因**，
  不要把两表结论混写。

---

## 4. 待 codex 做的事

1. 在 `main_arxiv_v1.tex` 现有 Vision-OPD6K 表之后，新建平行表（建议 `\label{tab:vopd6k_native}`），
   列 = Model / Method / V\* / HR4K / HR8K / ZoomBench，行结构与现表一致（两底座 × 三方法）；
2. caption 写明 §0 的口径差异 + 「不可与 7-bench 表跨表相减」+ Zoom 用 `zoombench.json`；
3. 待跑格用 TBD 占位，Q3-VL-4B base 三格标 pending re-run；
4. 三个在跑 job 出分后由 devbox 回填本文件，codex 据此更新 tex。
