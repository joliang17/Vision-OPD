# CS-OPSD Paper — 当前状态快照 (2026-07-24)

新一轮改进前的完整整理。重跑实验/更新 paper 时对照此文件。Overleaf 最新 commit 见 `git log`。

## 1. 口径约定（新一轮务必沿用，否则数字对不上主表）
- **7-bench** = BLINK / MMStar / VStarBench / MathVista_MINI / HRBench4K / HRBench8K / HallusionBench三均
- **HR** = `cycle=Average × type=all`（不是 cycle0）
- **Hallu** = (aAcc+fAcc+qAcc)/3
- **Acc** = 7 项算术平均
- 主线 = **Qwen3-VL-2B × virl39k × unfiltered × uniform × step90**；**ours(2B) = 67.04**
- ⚠️ 主表 ours 和 α/β 消融**都是 unfiltered**（同口径，filter 敏感性打平，别再纠结 filtered/unfiltered）

## 2. Paper 图表 → 数据源 → 生成方式
| Paper 元素 | 文件 | 数据源 | 生成 |
|---|---|---|---|
| Fig1 motivation | `Figure1_v2.pdf` | 用户 | 手工 |
| Fig2 framework | `method_pipeline_v2.pdf` | 用户 | 手工 |
| Table1 主表 | `tab:main` | paper_notes / paper_handoff_values_20260722.md | 手填 |
| Fig3 ablations | `combined_ablation.pdf` (`fig:ablations`) | ↓ §4 | `paper_figures.py::fig_combined` |
| Table divergence | `tab:divergence-direction` | paper_notes §14 | 手填 |
| Table control-img | `tab:control-image-ablation` | paper_notes §12 | 手填 |
| Table anchor | `tab:anchor-ablation` | paper_notes §12 | 手填 |
| Fig4 step-curve(三连) | `step_curve_row.pdf` (`fig:step-curve`) | fce_curves_20260723 CSV | `paper_figures.py::fig_stepcurve_row` |
| Fig5 case study | `casestudy_mathvista_163.png` (`fig:casestudy`) | MathVista idx163 真实输出（tex 内联文本） | 手工 |
| Fig6 contrast-vis(VDH) | `visual_token_v2.pdf` (`fig:contrast-vis`) | vdh_out/ (MMStar_282) | 用户 web 端拼 HTML 截图 |

## 3. 图生成脚本（一键全出）
`conda activate qwen35 && python3 Vision-OPD/scripts/paper_figures.py` → 生成全部 7 个 PDF 到 `overleaf-paper/figures/`：
- `fig_combined` → combined_ablation.pdf（**paper 用这个**，3 panel：β / α / anchor-drift）
- `fig_stepcurve_row` → step_curve_row.pdf（**paper 用这个**，3 panel：7bench / MMStar / MathVista）
- 单独版（备用，paper 未直接用）：`fig_beta`→beta_collapse, `fig_alpha`→alpha_curve, `fig_drift`→anchor_drift, `fig_stepcurve`→step_curve(两连), `fig_stepcurve_7bench`→step_curve_7bench
- 样式：C_OURS=#2E5E8C(蓝 o-, "ours") / C_ALT=#B4452F(红 s-, "OPSD")；全局 rcParams 统一

### 消融图当前数据（改数据就改这些函数）
- **β panel** (fig_combined a / fig_beta)：step [30,60,90,120,150]；β=0=[66.22,65.67,63.19,60.84,55.61]（崩溃）；β=0.1 ours=[66.22,65.92,67.04,65.91,65.38]（平稳）。源 `beta0_vs_FC1.csv`
- **α panel** (fig_combined b / fig_alpha)：α=[0,0.5,1.0,1.25,1.5,2.0]=[64.31,63.56,67.04,67.04,66.14,64.32]。**α=1.0 用主表 67.04**，其余重判后（paper_notes §12）
- **anchor-drift panel** (fig_combined c / fig_drift)：step [10,30,50,70,90]，no-anchor CJK% vs anchor 的差值

## 4. step curve 数据（fce_curves_20260723/）
- `FC1_ours.csv` / `FC4_baseline.csv`：15 步(10–150)，列 = step,Acc_7bench,BLINK,MMStar,VStar,MathVista,HRBench4K,HRBench8K,Hallu_3mean
- **step10 最新值**：FC1 65.51 / FC4 64.42（几经修正 57.96→64.37→65.51，主要是 HR8K + Hallu 补跑）
- `beta0_vs_FC1.csv`：β=0 vs FC1 崩溃对照，5 点(30–150)
- README.md 记口径/血缘/修复历史

## 5. VDH（visual dependency highlight）
- 脚本：`visual_dependency_highlight.py`（GPU 算 per-token Δ=logp(img)−logp(black)，批量走 manifest）+ `render_vdh.py`
- 数据：`docs/vdh_out/*.json`（13 样本），`docs/vdh_manifest.jsonl`，assets 在 `scripts/assets/vdh_batch/`
- paper 最终用：`visual_token_v2.pdf`（用户 web 端，MMStar_282，四行 Base/OPSD/Ours/Additional contrast）
- 详见 paper_notes §15

## 6. 关键 eval 目录（重跑时找旧结果）
- ours(contrast)：`cons_qwen35_virl39k_step90_seedA`（各 benchmark judge xlsx 全）、`contrast_standard_virl39k_90step_server`(=forward,7-bench 67.07)
- OPSD：`answerhint_qwen35_unfiltered_step90`
- base：`vanilla_qwen35_2b_nothink`
- divergence：reverse=`contrast_reversekl_2b_virl39k_step90`(64.77)，JSD=`outputs_api_server/jsd_2b_virl39k_90step_step90_eval`(66.25)，forward=上面 67.07
- VA-OPD(P35，**未完成**)：`vaopd_grouped_forward/rollout_step62`、`ra_vad_8b2b_vaopd_step62` — **都缺 HallusionBench，7-bench 算不全**，要用需补 Hallu eval

## 7. 本 session 主要改动（重跑对照基线）
- 主表 Qwen3.5 三底座 base → **no-think** 口径（ours 优势变 +2.9/+2.83/+4.27）
- §Divergence 新增消融表（forward 67.07 > JSD 66.25 > reverse 64.77）
- §Training Dynamics 三连 step curve（7bench/MMStar/MathVista，15 步）
- β 消融 5 点、α 消融 α=1.0 用主表 67.04
- 主表/anchor/control/divergence 全部统一到新 7-bench 口径

## 8. 已知未决 / 待清理
- VA-OPD(P35) 缺 Hallu，未纳入 paper
- `casestudy_vdh_mmstar282.png`（我的占位）还在 git，paper 实际用 `visual_token_v2.pdf`，占位可 `git rm`
- 单独版 step_curve.pdf / step_curve_7bench.pdf 在 figures/ 但 paper 未直接引用（三连版替代），留着无害
