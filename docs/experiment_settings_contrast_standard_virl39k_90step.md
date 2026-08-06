# 实验设置详录：contrast-标准 × virl39k-filtered（90步）

> ⚠️ **本文档记录的是 2026-07-13 那一次特定 run（filtered 数据 + RA-VAD 权重启用），不是 paper 主线配置。**
> paper 主线见 `docs/paper_state_snapshot_20260724.md:10`：**Qwen3-VL-2B × virl39k × unfiltered × uniform × step90，ours(2B)=67.04**。
> 两处关键差异：主线用 **unfiltered** 数据（本文档 §2 是 filtered 14,002 行），且 **`ra_uniform_weight=True`（权重关闭）**
> （本文档 §4.1 写的是 `False`/权重启用）。写 paper 时以主线快照为准，本文档仅作该次 run 的实现细节参考
> （公式推导、代码行号对照仍然有效）。—— 2026-08-05 核实并加注
>
> 对应 checkpoint：`checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-virl39k-filtered-90step/`。
> 全部公式与代码逐行核对过（关键实现：`verl/trainer/ppo/ra_vad.py`、`verl/workers/actor/dp_actor.py`）；另有 Codex 独立审查（结论见文末附录）。
> 训练命令（2026-07-13，trial 301638440，4×B200）：
> ```bash
> EXPERIMENT_NAME=Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-virl39k-filtered-90step \
> ANSWER_VAL_TRAIN_FILE=data/virl39k_train_noimg_filtered_1img.parquet \
> TRAIN_BATCH_SIZE=32 MAX_PROMPT_LENGTH=6144 \
> bash scripts/run_experiment_contrast_standard.sh \
>   data.filter_overlong_prompts=True trainer.total_training_steps=90
> ```

## 1. 总览

**方法一句话**：纯自蒸馏（无外部 teacher、无 ground-truth 标签、无 RL reward），student 的 rollout 由 EMA teacher 在两种条件下打分（hi=真实图像，ctrl=同尺寸黑图），二者的逐 token logprob 差既用来构造**对比锐化的蒸馏 target**（机制核心），也用来构造**逐 token 权重**（加权系数）。Loss 是加权的全词表 KL 蒸馏。

**训练范式**：on-policy——每步先由当前 student 采样 rollout，teacher 只对这些 rollout 做教师强制（teacher-forcing）打分，不生成。

## 2. 模型与数据

| 项 | 值 |
|---|---|
| Base 模型 | Qwen3-VL-2B-Instruct（HF snapshot 89644892…） |
| Teacher | student 权重的 EMA 拷贝（`teacher_regularization=ema`，`teacher_update_rate=0.05`，即每步 θ_T ← 0.95·θ_T + 0.05·θ_S） |
| 训练数据 | `virl39k_train_noimg_filtered_1img.parquet`：ViRL-39K 过滤版，再离线剔除多图样本后的**单图子集，14,002 条**（原始 filtered 版 14,861 条中 859 条含 2–8 图；多图样本在全词表蒸馏 backward 时触发确定性 OOM，故剔除；`data.filter_overlong_prompts=True` 再丢弃 >6144 token 的极端长样本） |
| 数据内容 | geometry/chart 为主的数学 MCQ（含图） |

## 3. 训练超参

| 项 | 值 | 备注 |
|---|---|---|
| GPU | 4 × B200 (183GB) | FSDP，param/optimizer offload 开启 |
| total_training_steps | **90**（硬截断） | 1 epoch 本为 ~437 步；437 步长训已证实训崩（详见 compare 文档），90 步在安全区内 |
| train_batch_size | 32 prompts/step | |
| rollout.n | 8 | 每 prompt 采 8 条 rollout → 每步 256 条 |
| lr | 2e-6（warmup 10 步，constant） | AdamW |
| ppo_mini_batch_size | 96 | |
| max_prompt_length / max_response_length | 6144 / 1024 | rollout `max_model_len`=7168 与 actor `ppo_max_token_len_per_gpu`=7168 由同一派生值保证一致 |
| rollout 引擎 | vLLM（tp=1, gpu_mem_util=0.7） | rollout 采样为 vLLM 默认随机采样 |
| loss_mode | `vopd` | actor 总 loss = 蒸馏 loss（`reward_model.enable=False`、`use_kl_loss=False`，advantage 通道虽配置为 grpo 但无 reward 参与，pg_loss=vopd_loss） |
| rollout 重要性修正 | `rollout_correction.rollout_is=token`, threshold=2.0 | ⚠️ **配置了但对 RA-VAD 路径不生效**（2026-07-16 Codex 审计）：trainer 计算了 `rollout_is_weights`，但 `ra_kd_loss` 调用（dp_actor.py:1348）不接收该参数（非 RA fallback 路径才消费）。RA-VAD 实际 loss 无 IS 修正，论文勿 claim。 |
| save_freq / max_actor_ckpt_to_keep | 10 / 10 | 90 步 × 每 10 步存 = 9 个 checkpoint 全保留 |

## 4. 机制与公式

符号：给定一条 rollout 的第 t 个 response token，teacher 在两种条件下教师强制打分：
- **hi 条件**：真实图像 + 原问题 → 逐 token 对数概率 `lp_hi`（全词表，经温度 T 软化）
- **ctrl 条件**：同尺寸黑图 + 原问题 → `lp_ctrl`

温度：`ra_temperature = T = 2.0`（teacher/student 的全词表 log-probs 都先除以 T 再 log_softmax；loss 末端乘 T² 校正梯度尺度，标准 Hinton KD 惯例）。

### 4.1 逐 token 权重 w_t（`compute_ra_weights`, ra_vad.py:64-147）

五步流水线，全程 stop-gradient：

```
a_t   = (lp_hi − lp_ctrl)_t · mask_t                     # 原始视觉依赖信号(标量,取自本token的已实现token对数概率差)
a⁺_t  = relu(a_t − δ),  δ = ra_delta = 0                  # 只保留正部
â_t   = min(a⁺_t, Q_row(0.95))                            # 逐行(每条rollout)95分位裁剪, ra_clip_quantile=0.95
w̃_t   = â_t / mean_row(â | â>0)                           # 行内正值均值归一化; 正值token数 < ra_min_positive_tokens=1 时整行置0
g     = σ(margin/0.5) · σ(ra_answer/0.1)                  # 样本级门控(标量/行):
                                                          #   margin = mean_row(a_t)  (该rollout平均hi/ctrl差)
                                                          #   ra_answer = mean_row(w̃_t | 正值token)
w_t   = stopgrad(w̃_t · g) · mask_t
```

配置：`ra_uniform_weight=False`（流水线启用。消融：置 True 时 w_t=mask_t，7-bench 掉 1.68pp，证明该权重是必要成分——见 compare 文档第十四轮）。

> ⚠️ **仅适用于本次 run。paper 主线相反：`ra_uniform_weight=True`（权重关闭，w_t=mask_t）**，见
> `paper_state_snapshot_20260724.md:10`。上面那句「该权重是必要成分」是 filtered 数据上的消融结论，
> 主线换到 unfiltered 后并未沿用。—— 2026-08-05 加注

### 4.2 对比锐化 target（`build_contrast_target`, ra_vad.py:344-419）

```
tilted_w = lp_hi(w) + α · (lp_hi(w) − lp_ctrl(w))          # 对全词表每个候选token w; α = ra_contrast_alpha = 1.0
tilted_w = lp_hi(w)                    if w ∈ E            # E = {<|endoftext|>=151643, <|im_end|>=151645}: 终止token的"tilt豁免"
                                                           #   (预检发现裸倾斜会抬高终止token导致回答变短; 注意E中token并未被移出
                                                           #    target支撑集——它们保留原lp_hi打分,仍受下一行plausibility mask约束)
tilted_w = −∞                          if lp_hi(w) < max_w' lp_hi(w') + log β    # β-plausibility mask, β = ra_contrast_beta = 0.1
target   = log_softmax(tilted)  (clamp_min −1e4, 防 0·−∞=NaN)
```

数学上等价于几何倾斜 `p_hi^{1+α} / p_ctrl^{α}`（支撑集限制在 hi 的 plausibility 集内）——把推理期 contrastive decoding 的操作蒸馏进训练。**注意**：此处 `lp_hi/lp_ctrl` 是温度 T=2 软化后的分布（forward 时 `logits /= T`，dp_actor.py:650），即 contrast 倾斜作用在 softened 分布上，论文公式需写明。

**标准版特有**：`ra_contrast_gate_positive_only=False`——所有位置都做倾斜（保守版=True 时只在 w_t>0 的位置倾斜，其余位置 target 退化为 p_hi、梯度≈0）。target 整体 detach（不反传 teacher）。

### 4.3 损失（`ra_kd_loss` + `token_divergence`, ra_vad.py:192-219, 420-500）

```
KL_t = KL( target_t ‖ p_student,t )    # forward KL(divergence_alpha=0): F.kl_div(student_logprobs, target, log_target=True)
                                       # 全词表求和, × T² (T=2)
L    = Σ_t w_t · KL_t / Σ_t w_t        # weighting_mode="continuous", 权重归一化在batch粒度
```

即 loss = **contrast target 上的加权 forward-KL 全词表蒸馏**（`full_logit_distillation=True`，`distillation_topk=null`——RA-VAD 硬性要求全词表，传 topk 会直接 ValueError）。

### 4.4 与保守（conservative）配置的差异

| 配置项 | 标准（本实验） | 保守 |
|---|---|---|
| α（倾斜强度） | **1.0** | 0.5 |
| position_gate | **关**（全位置倾斜） | 开（仅 w_t>0 位置） |
| 其余（β、E、ctrl、权重流水线） | 相同 | 相同 |

## 5. Rollout / teacher 构造细节

- rollout 生成时 prompt 用**真实图像+原问题**（ctrl 只存在于 teacher 打分侧，不影响采样）
- teacher hi/ctrl 两次 forward 用同一条 rollout 的 token 序列教师强制；ctrl 的黑图与原图同尺寸（`_make_black_images_like`），消除"有没有图"这个 meta 信号的混杂
- `teacher_always_on=True`，无 GRPO fallback（`policy_fallback_fraction=0` 全程）
- chat template：`chat_templates/perception_chat_template_qwen35.jinja`（训练与该系列所有对照组一致）

## 6. 评测配置（本 checkpoint 已发表数字的口径）

| 项 | 值 |
|---|---|
| 主口径 | VLMEvalKit `BACKEND=vllm_server`，temp=0，top_p=0.8，top_k=20，presence_penalty=1.5，max_new_tokens=4096 |
| Judge | Azure `gpt-5.4-mini-2026-03-17` |
| ZoomBench | 原生管线（infer.py→judge_qwenlm.py(v3-fixed)→cal_acc.py），temp=0，max_tokens=8192，外部 GPT judge |
| 主要结果（step90） | 7-bench 70.68（BLINK 58.18 / MMStar 63.87 / MMBench 78.18 / VStar 78.01 / MathVista 67.00 / HRBench4K 76.38 / HRBench8K 73.13），POPE 88.36，HallusionBench aAcc/fAcc/qAcc 68.56/44.51/45.71，ZoomBench 41.89 |
| 注意 | 与 Vision-OPD 论文官方口径（temp0.7 + max_pixels=16.7M + 官方judge）不同——官方口径复刻实验（2026-07-15）确认 HRBench 系列在官方口径下会高 ~5pp，跨口径比较时需注意 |

## 7. 已知边界与负结果（论文需如实呈现）

1. **训练长度安全区**：同配置 437 步训崩（7-bench 58.31，无限复读+中英混杂，崩溃始于 step 150-200）；90 步健康。机制解释：contrast target 相对 EMA teacher 现算，student→teacher→target 形成自增强回路，长训下失控（详见报告"训练动态"）
2. **步数非单调**：step30/60/90 = 69.62/69.61/70.68（本实验单调升，但保守版两条线均 step60 见顶回落），"90步最优"不成立，只是"90步安全"
3. **WeMath 负增益**（2026-07-15 math suite）：MathVista +7.7pp 的同时 WeMath Strict −2.4pp，伴随输出显著变长（超长回答占比 32.9% vs base 7.6%）——对"严格多步数学推理"类指标存在副作用

---
## 附录：Codex 独立配置审查结论（2026-07-15，gpt-5.3-codex）

**总判定：本次已完成的 run 配置正确**——全部 hydra override 生效无静默覆盖（参数顺序:主脚本固定参数→black分支→answer-val→用户参数最后覆盖，run_vision_opd_ra_vad.sh:409）、contrast 公式实现与文档一致（tilt ra_vad.py:378 / β-mask :386 / normalize :392）、外层 RA 权重未被 target 替代（weights=ra_weights*loss_mask，:445）、90步截断与 save_freq=10/keep=10 无交互坑（step90 是最后一步无条件保存，9 个 checkpoint 全保留且 shard 完整）。

**需要在论文/复现说明中披露的 4 项**：
1. **⚠️ 复现风险：`ANSWER_VAL_TRAIN_FILE` 可能被脚本静默重写**（run_vision_opd_ra_vad.sh:137-140,165）——validation split 初始化逻辑在特定 mtime/文件缺失条件下会用仓库 `train.parquet` 的内容覆盖用户传入的外部训练文件。本次 run 经日志核实未触发（实际训练文件确为 virl39k），但**复现命令建议显式关闭自动 split 或先修脚本**。
2. **措辞**：`exclude_token_ids` 是"tilt 豁免"不是"从 target 删除"——终止 token 保留原 lp_hi 打分、仍在支撑集内（受 β-mask 约束）。写"termination-token tilt exemption"，不要写"exclusion"。
3. **`resume_mode=auto` 是默认值**——同 EXPERIMENT_NAME 重跑会自动续训而非从头开始，复现命令建议加 `trainer.resume_mode=disable` 或保证输出目录全新。
4. **black ctrl 的准确描述**：控制图保留原图尺寸/图像占位符/视觉 token 数量，仅内容替换为纯黑 RGB（ray_trainer.py:860,1441）——是"内容抹除"而非"无图"，与 noimg ctrl 有本质区别。

**Codex 给出的论文可复现描述（英文，可直接改写使用）**：
> Qwen3-VL-2B-Instruct, black-image control, full-vocabulary forward-KL self-distillation with temperature T=2 and T² scaling. The target is contrast-sharpened using α=1.0, restricted to the p_hi ≥ 0.1·max p_hi support. `<|endoftext|>` and `<|im_end|>` retain their untilted high-image logits. The resulting KL remains weighted by detached RA-VAD token weights computed from the teacher-forced high-image/control chosen-token log-probability gap, with ReLU, per-row 95th-percentile clipping, positive-mean normalization, and sample gating. EMA teacher update rate is 0.05. Training uses batch size 32, rollout n=8, LR 2×10⁻⁶, 10-step warmup, maximum prompt/response lengths 6144/1024, and stops after exactly 90 optimizer iterations.
