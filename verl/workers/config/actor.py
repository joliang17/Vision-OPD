# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler.config import ProfilerConfig

from .engine import FSDPEngineConfig, McoreEngineConfig
from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = [
    "SelfDistillationConfig",
    "PolicyLossConfig",
    "RouterReplayConfig",
    "ActorConfig",
    "FSDPActorConfig",
    "McoreActorConfig",
]


@dataclass
class SelfDistillationConfig(BaseConfig):
    """Configuration for self-distillation loss.

    Args:
        Distillation is enabled when policy_loss.loss_mode is "vopd".
        full_logit_distillation (bool): Whether to use full-logit KL distillation.
        alpha (float): KL interpolation coefficient. 0.0=forward KL, 1.0=reverse KL, in-between=JSD.
        gamma (float): Weight applied to the SDPO loss.
        success_reward_threshold (float): Minimum sequence reward to be considered successful.
        teacher_regularization (str): Teacher regularization mode. Options: "ema", "trust-region", "progressive".
        teacher_update_rate (float): EMA update rate for teacher weights, or trust-region mixing coefficient.
        teacher_update_interval (Optional[int]): Hard-sync the teacher to the current student every N actor updates
            when teacher_regularization="progressive".
        distillation_topk (Optional[int]): If set, use top-k logits for distillation.
        distillation_add_tail (bool): Whether to add a tail bucket for top-k distillation.
        max_reprompt_len (int): Maximum length of the reprompted prompt.
        reprompt_truncation (str): Truncation method for the reprompted prompt (recommended to use "right" or "error").
        dont_reprompt_on_self_success (bool): Whether to not reprompt on self-success.
        remove_thinking_from_demonstration (bool): Whether to remove <think>...</think> tags from successful demonstrations before reprompting.
        is_clip (Optional[float]): Clip value for distillation IS ratio; None disables IS weighting.
        reprompt_template (str): Template for reprompting. Uses {prompt}, {solution}, {feedback} placeholders.
        solution_template (str): Template for formatting solution section. Uses {successful_previous_attempt} placeholder.
        feedback_template (str): Template for formatting feedback section. Uses {feedback_raw} placeholder.
        include_environment_feedback (bool): Whether to include environment feedback in reprompting for wrong attempts.
        environment_feedback_only_without_solution (bool): If True, only use feedback when no solution is available (ignore feedback when solution exists).
        reprompt_template_feedback (str): Template for reprompting with feedback but no solution.
        reprompt_template_feedback_solution (str): Template for reprompting with both feedback and solution.
        teacher_always_on (bool): Whether to distill every sample directly from a teacher input instead of selecting successful samples by reward.
        teacher_model_source (str): Teacher source. Options: "legacy", "current" or "fixed".
        teacher_model_path (Optional[str]): Fixed teacher model path when teacher_model_source="fixed".
        save_ema_teacher_checkpoint (bool): Save the EMA teacher weights alongside actor checkpoints.
        teacher_image_key (Optional[str]): Dataset column holding teacher-side images for multimodal distillation.
        fallback_to_policy_loss_on_missing_teacher (bool): When teacher_always_on=True, fall back to vanilla
            policy loss for samples whose teacher_image_key column is empty.
        log_prob_dump_dir (Optional[str]): Optional directory used to dump student/teacher log-prob tensors for each step.
    """

    full_logit_distillation: bool = True
    alpha: float = 0.0
    gamma: float = 1.0
    success_reward_threshold: float = 1.0
    teacher_regularization: str = "ema"
    teacher_update_rate: float = 0.05
    teacher_update_interval: Optional[int] = None
    distillation_topk: Optional[int] = None
    distillation_add_tail: bool = True
    max_reprompt_len: int = 10240
    reprompt_truncation: str = "right"
    dont_reprompt_on_self_success: bool = False
    remove_thinking_from_demonstration: bool = False
    is_clip: Optional[float] = None
    reprompt_template: str = (
        "{prompt}{solution}{feedback}\n\n"
        "Correctly solve the original question.\n"
    )
    solution_template: str = (
        "\n"
        "Correct solution:\n\n"
        "{successful_previous_attempt}\n\n"
    )
    feedback_template: str = (
        "\n"
        "The following is feedback from your unsuccessful earlier attempt:\n\n"
        "{feedback_raw}\n\n"
    )
    include_environment_feedback: bool = False
    environment_feedback_only_without_solution: bool = False
    teacher_always_on: bool = False
    teacher_model_source: str = "legacy"
    teacher_model_path: Optional[str] = None
    save_ema_teacher_checkpoint: bool = False
    teacher_image_key: Optional[str] = None
    teacher_prompt_mode: Optional[str] = None
    ra_vad: bool = False
    ra_ctrl_mode: str = "none"
    ra_ctrl_image_key: Optional[str] = None
    ra_generic_prompt: str = "Describe this image in detail."
    ra_delta: float = 0.0
    ra_clip_quantile: float = 0.95
    ra_min_positive_tokens: int = 1
    ra_temperature: float = 2.0
    ra_uniform_weight: bool = False
    ra_no_sample_gate: bool = False
    ra_margin_scale: float = 0.5
    ra_answer_scale: float = 0.1
    ra_weighting_mode: str = "continuous"
    ra_shuffle_seed: int = 12345
    # Source of the per-token relevance signal fed into compute_ra_weights:
    # "logprob" (default) = logp_hi - logp_ctrl teacher-forced gap;
    # "attention" = attention-to-image-token concentration, computed by a dedicated frozen,
    # unsharded (non-FSDP) copy of the base model loaded specifically for this — flash-attention
    # kernels never materialize attention weights, and extracting them from the live
    # FSDP-sharded teacher via ad-hoc eager-attention forwards ran into multiple FSDP1
    # summon_full_params/param_offload interaction bugs, see docs/compare_vaopd_0701.md Phase
    # 2-核心 第七/八轮. A separate frozen scorer sidesteps all of that and matches the offline
    # analysis methodology exactly (which also used a fixed base checkpoint, not a live teacher).
    ra_weight_source: str = "logprob"
    # Model path for the frozen attention scorer. Required when ra_weight_source="attention".
    # Typically the same base checkpoint used to initialize training (see actor.yaml default,
    # which interpolates actor_rollout_ref.model.path).
    ra_attention_scorer_model_path: Optional[str] = None
    # Qwen3-VL image placeholder token id (config.image_token_id). Required when
    # ra_weight_source="attention" so the extractor knows which key positions are image tokens.
    ra_attention_image_token_id: Optional[int] = None
    # Only average the last N decoder layers' attention into the score (all layers are still
    # computed by output_attentions=True regardless — this only affects which are used, not
    # memory/compute). <=0 means use all layers.
    ra_attention_num_layers: int = 8
    # How many samples to run through the attention scorer per forward call. Larger values reduce
    # Python/kernel-launch overhead (fewer, better-utilized calls) but raise peak memory linearly
    # (the O(seq_len^2) attention matrices dominate). Start at 1, raise only after confirming
    # headroom for the training sequence lengths in use.
    ra_attention_scorer_batch_size: int = 1
    # Opt-in diagnostic: per-step dump of a few samples' (response token id, RA weight) pairs to
    # `ra_token_dump_dir`/<experiment_name>/<step>.rank<r>.pt, for tracking "which tokens get high
    # weight, and how does that change over training" without a separate offline analysis run.
    # Off by default. Decode with scripts/summarize_ra_token_dump.py.
    ra_token_dump_dir: Optional[str] = None
    ra_token_dump_max_samples: int = 4
    # Distillation target: "teacher" (default, plain EMA-teacher hi distribution) or "contrast"
    # (contrast-sharpened target softmax(lp_hi + α(lp_hi − lp_ctrl)) restricted to hi's
    # plausibility set — see verl/trainer/ppo/ra_vad.py build_contrast_target and
    # docs/compare_vaopd_0701.md Phase 2-核心 第十轮 for motivation/diagnostics).
    ra_target_mode: str = "teacher"
    # Contrast tilt strength α (offline pre-check: 0.5 conservative, 1.0 standard).
    ra_contrast_alpha: float = 1.0
    # Anchor coefficient on the plain lp_hi term: target = softmax(anchor_coef·lp_hi + α·(lp_hi−lp_ctrl)).
    # 1.0 (default) = current contrast target; 0.0 + α=1.0 = pure contrastive-decoding target (no anchor).
    ra_contrast_anchor_coef: float = 1.0
    # Plausibility threshold β: candidate set = {w : p_hi(w) ≥ β max p_hi}.
    ra_contrast_beta: float = 0.1
    # If True, only apply the tilt at positions with positive RA weight; elsewhere the target
    # stays the plain hi distribution (≈ zero gradient there).
    ra_contrast_gate_positive_only: bool = False
    # Vocab ids excluded from the tilt (keep plain lp_hi score), e.g. <|im_end|>/<|endoftext|> —
    # the offline pre-check caught the raw tilt boosting early-termination tokens.
    ra_contrast_exclude_token_ids: Optional[list[int]] = None
    ra_vaopd_pv: float = 0.2
    ra_vaopd_lambda: float = 0.5
    ra_rollout_reweight: bool = False
    ra_rollout_tau: float = 1.0
    ra_divergence_alpha: float = 0.0
    # Visual-token budgets for the offline degrade control (R_res).
    # downsample target area = ra_ctrl_low_tokens * 28 * 28 (Qwen3-VL patch=28),
    # matching unsup-opsd/ra_vad `_LOW_TOKENS/_UP_TOKENS`.
    ra_ctrl_low_tokens: int = 256
    ra_ctrl_up_tokens: int = 1536
    answer_hint_template: str = (
        "\n\nHere is a reference solution to this problem:\n"
        "{answer}\n\n"
        "After understanding the reference solution, please try to solve this problem using your own approach below:\n"
    )
    fallback_to_policy_loss_on_missing_teacher: bool = False
    log_prob_dump_dir: Optional[str] = None

    def __post_init__(self):
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"self_distillation.alpha must be in [0,1], got {self.alpha}")
        if self.gamma < 0.0:
            raise ValueError(f"self_distillation.gamma must be non-negative, got {self.gamma}")
        valid_teacher_regularization = ["ema", "trust-region", "progressive"]
        if self.teacher_regularization not in valid_teacher_regularization:
            raise ValueError(
                "self_distillation.teacher_regularization must be one of "
                f"{valid_teacher_regularization}, got {self.teacher_regularization}"
            )
        if not 0.0 <= self.teacher_update_rate <= 1.0:
            raise ValueError(
                f"self_distillation.teacher_update_rate must be in [0,1], got {self.teacher_update_rate}"
            )
        if self.teacher_update_interval is not None and self.teacher_update_interval <= 0:
            raise ValueError(
                "self_distillation.teacher_update_interval must be a positive integer "
                f"when set, got {self.teacher_update_interval}"
            )
        if self.distillation_topk is not None and self.distillation_topk <= 0:
            raise ValueError(
                f"self_distillation.distillation_topk must be a positive integer, got {self.distillation_topk}"
            )
        if self.is_clip is not None and self.is_clip <= 0:
            raise ValueError(f"self_distillation.is_clip must be positive, got {self.is_clip}")
        if self.teacher_prompt_mode is not None and self.teacher_prompt_mode != "answer_hint":
            raise ValueError(
                f"self_distillation.teacher_prompt_mode must be None or 'answer_hint', got {self.teacher_prompt_mode}"
            )
        valid_ra_ctrl_modes = ["none", "degrade", "qvis", "noimg", "black", "gaussnoise"]
        if self.ra_ctrl_mode not in valid_ra_ctrl_modes:
            raise ValueError(
                f"self_distillation.ra_ctrl_mode must be one of {valid_ra_ctrl_modes}, got {self.ra_ctrl_mode}"
            )
        if self.ra_vad:
            if self.teacher_prompt_mode is not None:
                raise ValueError("self_distillation.ra_vad requires teacher_prompt_mode=None.")
            if self.ra_ctrl_mode == "none":
                raise ValueError("self_distillation.ra_vad requires ra_ctrl_mode != 'none'.")
            if self.ra_ctrl_mode in {"none", "degrade", "qvis", "black", "gaussnoise"} and not self.ra_ctrl_image_key:
                raise ValueError(
                    "self_distillation.ra_ctrl_image_key is required when ra_vad=True and "
                    f"ra_ctrl_mode={self.ra_ctrl_mode!r}."
                )
            if not 0.0 < self.ra_clip_quantile <= 1.0:
                raise ValueError(
                    f"self_distillation.ra_clip_quantile must be in (0,1], got {self.ra_clip_quantile}"
                )
            if self.ra_min_positive_tokens <= 0:
                raise ValueError(
                    "self_distillation.ra_min_positive_tokens must be positive, "
                    f"got {self.ra_min_positive_tokens}"
                )
            if self.ra_temperature <= 0:
                raise ValueError(f"self_distillation.ra_temperature must be positive, got {self.ra_temperature}")
            valid_ra_weighting_modes = ["continuous", "vaopd_grouped", "shuffled_control"]
            if self.ra_weighting_mode not in valid_ra_weighting_modes:
                raise ValueError(
                    "self_distillation.ra_weighting_mode must be one of "
                    f"{valid_ra_weighting_modes}, got {self.ra_weighting_mode}"
                )
            valid_ra_weight_sources = ["logprob", "attention"]
            if self.ra_weight_source not in valid_ra_weight_sources:
                raise ValueError(
                    "self_distillation.ra_weight_source must be one of "
                    f"{valid_ra_weight_sources}, got {self.ra_weight_source}"
                )
            if self.ra_weight_source == "attention":
                if self.ra_attention_image_token_id is None:
                    raise ValueError(
                        "self_distillation.ra_attention_image_token_id is required when "
                        "ra_weight_source='attention'."
                    )
                if not self.ra_attention_scorer_model_path:
                    raise ValueError(
                        "self_distillation.ra_attention_scorer_model_path is required when "
                        "ra_weight_source='attention'."
                    )
            valid_ra_target_modes = ["teacher", "contrast"]
            if self.ra_target_mode not in valid_ra_target_modes:
                raise ValueError(
                    "self_distillation.ra_target_mode must be one of "
                    f"{valid_ra_target_modes}, got {self.ra_target_mode}"
                )
            if self.ra_target_mode == "contrast":
                if not self.full_logit_distillation or self.distillation_topk is not None:
                    raise ValueError(
                        "self_distillation.ra_target_mode='contrast' requires "
                        "full_logit_distillation=True and distillation_topk=null (the target is "
                        "built over the full vocabulary of both the hi and ctrl branches)."
                    )
                # alpha=0 disables the contrast tilt entirely (tilted = lp_hi), leaving a fully
                # matched pure-EMA-teacher self-distillation target that keeps the identical β mask,
                # exclude-token, uniform-weight and forward-KL pipeline — this is the paper's
                # matched α=0 baseline (only variable vs ours is the tilt term). Numerically safe:
                # ra_vad.py build_contrast_target reduces to log_softmax(lp_hi over plausible set).
                # Negative alpha is still rejected (would anti-tilt toward the ctrl branch).
                if self.ra_contrast_alpha < 0:
                    raise ValueError(
                        f"self_distillation.ra_contrast_alpha must be non-negative, got {self.ra_contrast_alpha}"
                    )
                # beta=0 disables the plausibility mask entirely (log(0)=-inf -> all tokens pass),
                # numerically safe in ra_vad.py:387; allowed for the P13 no-mask ablation (2026-07-17).
                if not 0.0 <= self.ra_contrast_beta < 1.0:
                    raise ValueError(
                        f"self_distillation.ra_contrast_beta must be in [0,1), got {self.ra_contrast_beta}"
                    )
            if not 0.0 < self.ra_vaopd_pv <= 1.0:
                raise ValueError(f"self_distillation.ra_vaopd_pv must be in (0,1], got {self.ra_vaopd_pv}")
            if not 0.0 <= self.ra_vaopd_lambda <= 1.0:
                raise ValueError(
                    f"self_distillation.ra_vaopd_lambda must be in [0,1], got {self.ra_vaopd_lambda}"
                )
            if self.ra_rollout_tau <= 0:
                raise ValueError(f"self_distillation.ra_rollout_tau must be positive, got {self.ra_rollout_tau}")
            if not 0.0 <= self.ra_divergence_alpha <= 1.0:
                raise ValueError(
                    "self_distillation.ra_divergence_alpha must be in [0,1], "
                    f"got {self.ra_divergence_alpha}"
                )
            if self.ra_margin_scale <= 0:
                raise ValueError(f"self_distillation.ra_margin_scale must be positive, got {self.ra_margin_scale}")
            if self.ra_answer_scale <= 0:
                raise ValueError(f"self_distillation.ra_answer_scale must be positive, got {self.ra_answer_scale}")
            if self.ra_ctrl_low_tokens <= 0:
                raise ValueError(
                    f"self_distillation.ra_ctrl_low_tokens must be positive, got {self.ra_ctrl_low_tokens}"
                )
            if self.ra_ctrl_up_tokens <= 0:
                raise ValueError(
                    f"self_distillation.ra_ctrl_up_tokens must be positive, got {self.ra_ctrl_up_tokens}"
                )
        if self.teacher_always_on and not self.teacher_image_key and self.teacher_prompt_mode != "answer_hint":
            raise ValueError(
                "self_distillation.teacher_image_key is required when teacher_always_on=True "
                "(unless teacher_prompt_mode='answer_hint')"
            )
        valid_teacher_model_source = ["legacy", "current", "fixed"]
        if self.teacher_model_source not in valid_teacher_model_source:
            raise ValueError(
                "self_distillation.teacher_model_source must be one of "
                f"{valid_teacher_model_source}, got {self.teacher_model_source}"
            )
        if self.teacher_model_source == "fixed" and not self.teacher_model_path:
            raise ValueError("self_distillation.teacher_model_path is required when teacher_model_source='fixed'")
        if self.teacher_regularization == "progressive":
            if self.teacher_model_source != "legacy":
                raise ValueError(
                    "self_distillation.teacher_regularization='progressive' requires "
                    "teacher_model_source='legacy'"
                )
            if self.teacher_update_interval is None:
                raise ValueError(
                    "self_distillation.teacher_update_interval is required when "
                    "teacher_regularization='progressive'"
                )


@dataclass
class RouterReplayConfig(BaseConfig):
    """Configuration for router replay in MoE models.

    This configuration controls the routing behavior for Mixture of Experts (MoE) models,
    allowing for deterministic training through route recording and replay.

    Args:
        mode (str): Router replay mode. Options: 'disabled', 'R2', 'R3'.
            - 'disabled': No router replay functionality
            - 'R2': Use Router Replay routing strategy
            - 'R3': Use Rollout Router Replay routing strategy
        record_file (Optional[str]): File path to save recorded routing decisions.
            Required when mode is 'record', 'R2', or 'R3'.
        replay_file (Optional[str]): File path to load recorded routing decisions for replay.
            Required when mode is 'replay'.
    """

    mode: str = "disabled"
    record_file: Optional[str] = None
    replay_file: Optional[str] = None

    def __post_init__(self):
        """Validate router replay configuration."""
        valid_modes = ["disabled", "R2", "R3"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid router_replay mode: {self.mode}. Must be one of {valid_modes}")


@dataclass
class PolicyLossConfig(BaseConfig):
    """Configuration for policy loss computation.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        loss_mode (str): Loss function mode. Options: 'vanilla', 'clip-cov', 'kl-cov', 'gpg', 'vopd'.
        clip_cov_ratio (float): Ratio of tokens to be clipped for clip-cov loss.
        clip_cov_lb (float): Lower bound for clip-cov loss.
        clip_cov_ub (float): Upper bound for clip-cov loss.
        kl_cov_ratio (float): Ratio of tokens to be applied KL penalty for kl-cov loss.
        ppo_kl_coef (float): KL divergence penalty coefficient.
    """

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1


@dataclass
class ActorConfig(BaseConfig):
    """Configuration for actor model training.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy. Must be specified.
        ppo_mini_batch_size (int): Mini-batch size for PPO training.
        ppo_micro_batch_size (Optional[int]): Micro-batch size for PPO training.
            If None, uses ppo_micro_batch_size_per_gpu.
        ppo_micro_batch_size_per_gpu (Optional[int]): Micro-batch size per GPU for PPO training.
        use_dynamic_bsz (bool): Whether to use dynamic batch sizing.
        ppo_max_token_len_per_gpu (int): Maximum token length per GPU for PPO training.
        clip_ratio (float): PPO clipping ratio for policy loss.
        clip_ratio_low (float): Lower bound for PPO clipping ratio.
        clip_ratio_high (float): Upper bound for PPO clipping ratio.
        policy_loss (PolicyLossConfig): Configuration for policy loss computation.
        clip_ratio_c (float): Clipping ratio for critic loss.
        loss_agg_mode (str): Loss aggregation mode. Options: 'token-mean', 'sample-mean'.
        loss_scale_factor (Optional[int]): Scale factor for 'seq-mean-token-sum-norm' loss aggregation mode.
            If None, uses response_length. Set to a constant to ensure consistent normalization.
        entropy_coeff (float): Entropy coefficient for regularization.
        tau_pos (float): Positive tau for SAPO smoothing (>= 1.0 keeps rewards stable).
        tau_neg (float): Negative tau for SAPO smoothing (> tau_pos for asymmetry).
        use_kl_loss (bool): Whether to use KL divergence loss.
        use_torch_compile (bool): Whether to use torch.compile for optimization.
        kl_loss_coef (float): KL divergence loss coefficient.
        kl_loss_type (str): Type of KL loss to use.
        ppo_epochs (int): Number of PPO epochs per training step.
        shuffle (bool): Whether to shuffle data during training.
        checkpoint (CheckpointConfig): Configuration for checkpointing.
        optim (OptimizerConfig): Configuration for optimizer.
        use_fused_kernels (bool): Whether to use custom fused kernels (e.g., FlashAttention, fused MLP).
        data_loader_seed (int): Seed for data loader. If None, uses global seed.
        router_replay (RouterReplayConfig): Configuration for router replay in MoE models.
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "ppo_infer_micro_batch_size_per_gpu",
        "engine",
        "model_config",
    }

    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None  # deprecate
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    ppo_infer_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    freeze_vision_tower: bool = False
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    loss_scale_factor: Optional[int] = None
    entropy_coeff: float = 0
    tau_pos: float = 1.0
    tau_neg: float = 1.05
    calculate_entropy: bool = False
    use_kl_loss: bool = False
    # Whether to enable PrefixGrouper-based shared-prefix forward
    use_prefix_grouper: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    data_loader_seed: int = 1
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    rollout_n: int = MISSING  # must be override by sampling config
    model_config: HFModelConfig = field(default_factory=BaseConfig)
    router_replay: RouterReplayConfig = field(default_factory=RouterReplayConfig)
    self_distillation: SelfDistillationConfig = field(default_factory=SelfDistillationConfig)

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate actor configuration parameters."""
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING
        if not self.use_dynamic_bsz:
            if self.ppo_micro_batch_size is not None and self.ppo_micro_batch_size_per_gpu is not None:
                raise ValueError(
                    "[actor] You have set both 'actor.ppo_micro_batch_size' AND 'actor.ppo_micro_batch_size_per_gpu'. "
                    "Please remove 'actor.ppo_micro_batch_size' because only '*_ppo_micro_batch_size_per_gpu' is "
                    "supported (the former is deprecated)."
                )
            else:
                assert not (self.ppo_micro_batch_size is None and self.ppo_micro_batch_size_per_gpu is None), (
                    "[actor] Please set at least one of 'actor.ppo_micro_batch_size' or "
                    "'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled."
                )

        valid_loss_agg_modes = [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ]
        if self.loss_agg_mode not in valid_loss_agg_modes:
            raise ValueError(f"Invalid loss_agg_mode: {self.loss_agg_mode}")

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate actor configuration with runtime parameters."""
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"actor.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

            sp_size = getattr(self, "ulysses_sequence_parallel_size", 1)
            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options."""
        param = "ppo_micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreActorConfig(ActorConfig):
    """Configuration for Megatron actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'megatron' for Megatron parallelism.
        load_weight (bool): Whether to load model weights from checkpoint.
        megatron (dict[str, Any]): Configuration for Megatron parallelism settings.
        profile (dict[str, Any]): Configuration for profiling settings.
    """

    strategy: str = "megatron"
    load_weight: bool = True
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.megatron


@dataclass
class FSDPActorConfig(ActorConfig):
    """Configuration for FSDP actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'fsdp' for Fully Sharded Data Parallel.
        grad_clip (float): Gradient clipping threshold.
        ulysses_sequence_parallel_size (int): [DEPRECATED] Ulysses sequence parallel size for long sequences.
        entropy_from_logits_with_chunking (bool): Whether to compute entropy from logits
            with chunking for memory efficiency.
        entropy_checkpointing (bool): Whether to use gradient checkpointing for entropy computation.
        fsdp_config (dict[str, Any]): Configuration for FSDP settings.
        use_remove_padding (bool): Whether to remove padding tokens in inputs during training
    """

    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    use_remove_padding: bool = False
    use_rollout_log_probs: bool = False
    calculate_sum_pi_squared: bool = False
    sum_pi_squared_checkpointing: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.fsdp_config

        # backward compatibility
        if self.ulysses_sequence_parallel_size > 1:
            self.fsdp_config.ulysses_sequence_parallel_size = self.ulysses_sequence_parallel_size

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate FSDP actor configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size, model_config)

        if self.strategy in {"fsdp", "fsdp2"} and self.ulysses_sequence_parallel_size > 1:
            if model_config and not model_config.get("use_remove_padding", False):
                raise ValueError(
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )
