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

from collections.abc import Sequence
from typing import Any, Optional

import torch
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    denom = mask.sum(dim=dim).clamp(min=1.0)
    return (values * mask).sum(dim=dim) / denom


_SHUFFLE_GENERATORS: dict[tuple[int, str], torch.Generator] = {}


def _get_shuffle_generator(seed: int, device: torch.device) -> torch.Generator:
    """Return a persistent RNG for the shuffled_control ablation, seeded independently of the
    training RNG. Cached per (seed, device) so the shuffle sequence advances across calls within
    a run but is reproducible if the whole run is repeated with the same `ra_shuffle_seed`."""
    key = (int(seed), str(device))
    generator = _SHUFFLE_GENERATORS.get(key)
    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        _SHUFFLE_GENERATORS[key] = generator
    return generator


def _shuffle_weights_per_row(weights: torch.Tensor, mask: torch.Tensor, seed: int) -> torch.Tensor:
    """Randomly permute `weights` within each row, among positions where `mask > 0` only.

    Preserves the exact per-row multiset of weight values (same mean/quantiles); only the
    token<->weight correspondence is broken. This isolates "does the specific visual-relevance
    signal matter" from "does any non-uniform gradient reweighting help".
    """
    generator = _get_shuffle_generator(seed, weights.device)
    rand_vals = torch.rand(weights.shape, generator=generator, device=weights.device, dtype=weights.dtype)
    valid = mask > 0
    # Random order among valid positions; invalid positions sorted after all valid ones.
    perm_key = torch.where(valid, rand_vals, rand_vals + 2.0)
    perm_order = torch.argsort(perm_key, dim=-1)
    # Original left-to-right order of valid positions (stable), used as the scatter target so the
    # shuffle only reassigns *which* valid token gets which weight, not the set of positions touched.
    orig_order = torch.argsort((~valid).to(dtype=torch.long), dim=-1, stable=True)
    permuted_values = torch.gather(weights, dim=-1, index=perm_order)
    shuffled = torch.zeros_like(weights)
    shuffled.scatter_(dim=-1, index=orig_order, src=permuted_values)
    return shuffled


def compute_ra_weights(
    logp_hi: torch.Tensor,
    logp_ctrl: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    delta: float = 0.0,
    clip_quantile: float = 0.95,
    min_positive_tokens: int = 1,
    uniform_weight: bool = False,
    no_sample_gate: bool = False,
    margin_scale: float = 0.5,
    answer_scale: float = 0.1,
    ra_raw_override: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute stop-gradient RA token weights from a per-token relevance signal.

    By default the relevance signal is the paired teacher-forced logprob gap
    ``logp_hi - logp_ctrl`` (image-present vs image-absent/degraded). Pass
    ``ra_raw_override`` to swap in a different per-token relevance signal (e.g. an
    attention-to-image-token concentration score, see ``ra_weight_source="attention"``
    in the actor config) while reusing the same clip/normalize/gate pipeline —
    everything downstream of the raw signal (delta threshold, quantile clipping,
    per-row normalization, sample-level gate) is signal-agnostic.
    """
    mask = response_mask.to(dtype=logp_hi.dtype)
    default_raw = logp_hi - logp_ctrl
    if uniform_weight:
        weights = mask.detach()
        return weights, {
            "ra_vad/ra_raw_mean": _masked_mean(
                default_raw if ra_raw_override is None else ra_raw_override, mask
            ).mean().detach().item(),
            "ra_vad/ra_norm_mean": 1.0,
            "ra_vad/ra_weight_mean": _masked_mean(weights, mask).mean().detach().item(),
            "ra_vad/g_sample_mean": 1.0,
            "ra_vad/positive_token_fraction": mask.float().mean().detach().item(),
        }

    raw_signal = default_raw if ra_raw_override is None else ra_raw_override.to(dtype=logp_hi.dtype)
    ra_raw = raw_signal * mask
    ra_pos = torch.relu(ra_raw - float(delta)) * mask

    positive_mask = (ra_pos > 0).to(dtype=ra_pos.dtype) * mask
    positive_count = positive_mask.sum(dim=-1, keepdim=True)
    enough_positive = positive_count >= float(min_positive_tokens)

    if clip_quantile < 1.0:
        large = torch.finfo(ra_pos.dtype).max / 4
        masked_pos = ra_pos.masked_fill(positive_mask <= 0, large)
        sorted_pos, _ = torch.sort(masked_pos, dim=-1)
        quantile_idx = torch.clamp((positive_count * clip_quantile).ceil().long() - 1, min=0)
        quantile_idx = torch.minimum(quantile_idx, torch.full_like(quantile_idx, ra_pos.shape[-1] - 1))
        clip_value = sorted_pos.gather(dim=-1, index=quantile_idx).masked_fill(positive_count <= 0, 0.0)
        ra_pos = torch.minimum(ra_pos, clip_value)

    positive_mean = (ra_pos * positive_mask).sum(dim=-1, keepdim=True) / positive_count.clamp(min=1.0)
    ra_norm = torch.where(enough_positive, ra_pos / positive_mean.clamp(min=1e-8), torch.zeros_like(ra_pos))
    ra_norm = ra_norm * mask

    if no_sample_gate:
        g_sample = torch.ones((ra_norm.shape[0], 1), dtype=ra_norm.dtype, device=ra_norm.device)
    else:
        margin = _masked_mean(ra_raw, mask, dim=-1).unsqueeze(-1)
        ra_answer = (ra_norm * positive_mask).sum(dim=-1, keepdim=True) / positive_count.clamp(min=1.0)
        ra_answer = torch.where(enough_positive, ra_answer, torch.zeros_like(ra_answer))
        g_sample = torch.sigmoid(margin / float(margin_scale)) * torch.sigmoid(ra_answer / float(answer_scale))

    weights = (ra_norm * g_sample).detach() * mask
    valid_count = mask.sum().clamp(min=1.0)
    positive_fraction = positive_mask.sum() / valid_count
    weight_sum = weights.sum(dim=-1)
    active_sample_fraction = (weight_sum > 0).float().mean()

    metrics = {
        "ra_vad/ra_raw_mean": (ra_raw.sum() / valid_count).detach().item(),
        "ra_vad/ra_pos_mean": (ra_pos.sum() / valid_count).detach().item(),
        "ra_vad/ra_norm_mean": (ra_norm.sum() / valid_count).detach().item(),
        "ra_vad/ra_weight_mean": (weights.sum() / valid_count).detach().item(),
        "ra_vad/g_sample_mean": g_sample.mean().detach().item(),
        "ra_vad/positive_token_fraction": positive_fraction.detach().item(),
        "ra_vad/active_sample_fraction": active_sample_fraction.detach().item(),
    }
    return weights, metrics


def attention_to_image_score(
    attentions: Sequence[torch.Tensor],
    image_token_mask: torch.Tensor,
    response_positions: torch.Tensor,
) -> torch.Tensor:
    """Per-response-token attention-to-image-token concentration, averaged over layers/heads.

    Offline analysis (``scripts/analyze_ra_tokens.py`` / ``scripts/analyze_ra_attention.py``, see
    docs/compare_vaopd_0701.md Phase 2-核心 第七/八轮) found this signal ranks true visual-content
    tokens far above discourse-marker/opening-phrase tokens, unlike the ``logp_hi - logp_ctrl``
    signal which is dominated by the model's "do I have an image at all" opening-phrase reaction.
    Feed the output of this function into ``compute_ra_weights(..., ra_raw_override=...)`` to use
    it as the RA-VAD relevance signal (``ra_weight_source="attention"``).

    Args:
        attentions: non-empty sequence of per-layer post-softmax attention weights, each shaped
            (batch, heads, seq, seq) — i.e. the ``attentions`` tuple a HF model returns with
            ``output_attentions=True`` under eager (non-flash) attention. Every layer must share
            the same seq_len; callers may pass a subset of layers to bound memory/compute.
        image_token_mask: (batch, seq) bool/float mask, 1 at image-token key positions.
        response_positions: (batch, response_length) long tensor of the query index (in the
            same seq dimension as ``attentions``) that predicts each response token — see
            ``DataParallelPPOActor._build_response_positions``.

    Returns:
        (batch, response_length) float32 tensor, unnormalized (not yet clipped/normalized —
        pass through ``compute_ra_weights`` for that).
    """
    if not attentions:
        raise ValueError("attentions must be a non-empty sequence of per-layer attention tensors.")
    batch, response_length = response_positions.shape
    seq_len = attentions[0].shape[-1]
    device = response_positions.device
    img_mask_f = image_token_mask.to(dtype=torch.float32, device=device)  # (batch, seq)
    gather_index = response_positions.unsqueeze(-1).expand(-1, -1, seq_len)  # (batch, resp_len, seq)

    attn_sum = torch.zeros(batch, response_length, dtype=torch.float32, device=device)
    for layer_attn in attentions:
        head_mean = layer_attn.mean(dim=1).to(dtype=torch.float32, device=device)  # (batch, seq, seq)
        rows = torch.gather(head_mean, dim=1, index=gather_index)  # (batch, resp_len, seq)
        attn_sum += (rows * img_mask_f.unsqueeze(1)).sum(dim=-1)
    return attn_sum / len(attentions)


def token_divergence(
    teacher_all_log_probs: torch.Tensor,
    student_all_log_probs: torch.Tensor,
    *,
    alpha: float = 0.0,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Per-token full-vocab divergence: 0=forward KL, 1=reverse KL, intermediate=generalized JSD."""
    alpha = float(alpha)
    if alpha == 0.0:
        loss = F.kl_div(student_all_log_probs, teacher_all_log_probs, reduction="none", log_target=True)
    elif alpha == 1.0:
        loss = F.kl_div(teacher_all_log_probs, student_all_log_probs, reduction="none", log_target=True)
    else:
        alpha_t = torch.tensor(alpha, dtype=student_all_log_probs.dtype, device=student_all_log_probs.device)
        mixture_log_probs = torch.logsumexp(
            torch.stack(
                [
                    student_all_log_probs + torch.log1p(-alpha_t),
                    teacher_all_log_probs + torch.log(alpha_t),
                ]
            ),
            dim=0,
        )
        kl_teacher = F.kl_div(mixture_log_probs, teacher_all_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_all_log_probs, reduction="none", log_target=True)
        loss = torch.lerp(kl_student, kl_teacher, alpha_t)
    return loss.sum(dim=-1) * (float(temperature) ** 2)


def token_kl(
    teacher_all_log_probs: torch.Tensor,
    student_all_log_probs: torch.Tensor,
    *,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Per-token forward KL KL(p_teacher || p_student) over the full vocabulary."""
    return token_divergence(
        teacher_all_log_probs=teacher_all_log_probs,
        student_all_log_probs=student_all_log_probs,
        alpha=0.0,
        temperature=temperature,
    )


def _sample_rollout_weights(
    ra_raw: torch.Tensor,
    loss_mask: torch.Tensor,
    uids: Optional[Sequence[Any]],
    *,
    tau: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute VA-OPD rollout-level weights within the local micro-batch."""
    device = ra_raw.device
    dtype = ra_raw.dtype
    sample_weight = torch.ones(ra_raw.shape[0], dtype=dtype, device=device)
    metrics = {
        "ra_vad/rollout_reweight_group_count": 0.0,
        "ra_vad/rollout_reweight_complete_group_fraction": 0.0,
        "ra_vad/rollout_reweight_weight_mean": 1.0,
    }
    if uids is None:
        return sample_weight, metrics

    uid_list = list(uids)
    if len(uid_list) != ra_raw.shape[0]:
        return sample_weight, metrics

    sample_va = _masked_mean(ra_raw, loss_mask, dim=-1)
    groups: dict[Any, list[int]] = {}
    for idx, uid in enumerate(uid_list):
        groups.setdefault(uid, []).append(idx)

    complete_groups = 0
    for indices in groups.values():
        idx = torch.tensor(indices, dtype=torch.long, device=device)
        if len(indices) < 2:
            sample_weight[idx] = 1.0
            continue
        complete_groups += 1
        values = sample_va[idx]
        centered = values - values.mean()
        std = values.std(unbiased=False)
        if torch.isfinite(std).item() and std.item() > 1e-8:
            logits = centered / std / float(tau)
        else:
            logits = torch.zeros_like(values)
        sample_weight[idx] = torch.softmax(logits, dim=0) * len(indices)

    group_count = max(len(groups), 1)
    metrics = {
        "ra_vad/rollout_reweight_group_count": float(len(groups)),
        "ra_vad/rollout_reweight_complete_group_fraction": complete_groups / group_count,
        "ra_vad/rollout_reweight_weight_mean": sample_weight.detach().mean().item(),
    }
    return sample_weight, metrics


def _grouped_ra_loss(
    per_token_loss: torch.Tensor,
    ra_raw: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    pv: float,
    group_lambda: float,
    rollout_sample_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    dtype = per_token_loss.dtype
    mask = loss_mask.to(dtype=dtype)
    bsz, response_len = per_token_loss.shape
    high_mask = torch.zeros_like(mask)

    valid_counts = mask.sum(dim=-1).long()
    for row in range(bsz):
        valid_count = int(valid_counts[row].item())
        if valid_count <= 0:
            continue
        high_count = max(1, int(torch.ceil(valid_counts[row].float() * float(pv)).item()))
        scores = ra_raw[row].masked_fill(mask[row] <= 0, -torch.inf)
        top_idx = torch.topk(scores, k=min(high_count, response_len), dim=-1).indices
        high_mask[row, top_idx] = 1.0
    high_mask = high_mask * mask
    low_mask = (mask - high_mask).clamp(min=0.0)

    high_denom = high_mask.sum(dim=-1).clamp(min=1.0)
    low_denom = low_mask.sum(dim=-1).clamp(min=1.0)
    high_loss = (per_token_loss * high_mask).sum(dim=-1) / high_denom
    low_loss = (per_token_loss * low_mask).sum(dim=-1) / low_denom
    has_low = (low_mask.sum(dim=-1) > 0).to(dtype=dtype)
    sample_loss = float(group_lambda) * high_loss + (1.0 - float(group_lambda)) * low_loss * has_low
    sample_loss = torch.where(has_low > 0, sample_loss, high_loss)

    active_sample = (valid_counts > 0).to(dtype=dtype)
    if rollout_sample_weights is not None:
        active_sample = active_sample * rollout_sample_weights.to(dtype=dtype)
    denom = active_sample.sum().clamp(min=1.0)
    loss = (sample_loss * active_sample).sum() / denom

    valid_count = mask.sum().clamp(min=1.0)
    metrics = {
        "ra_vad/grouped_high_token_fraction": (high_mask.sum() / valid_count).detach().item(),
        "ra_vad/grouped_high_loss_mean": (
            (per_token_loss * high_mask).sum() / high_mask.sum().clamp(min=1.0)
        ).detach().item(),
        "ra_vad/grouped_low_loss_mean": (
            (per_token_loss * low_mask).sum() / low_mask.sum().clamp(min=1.0)
        ).detach().item(),
    }
    return loss, metrics


@torch.no_grad()
def build_contrast_target(
    teacher_hi_all_log_probs: torch.Tensor,
    teacher_ctrl_all_log_probs: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 0.1,
    anchor_coef: float = 1.0,
    position_gate: Optional[torch.Tensor] = None,
    exclude_token_ids: Optional[Sequence[int]] = None,
    metric_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Contrast-sharpened distillation target: softmax(anchor_coef·lp_hi + α (lp_hi − lp_ctrl))
    restricted to the plausibility set {w : p_hi(w) ≥ β max p_hi}. anchor_coef=1 (default) keeps the
    lp_hi anchor (current design); anchor_coef=0 + α=1 gives softmax(lp_hi − lp_ctrl), the pure
    contrastive-decoding target (ablation of the anchor).

    Motivation (docs/compare_vaopd_0701.md Phase 2-核心 第十轮): the identical-input EMA teacher
    differs from the student by ~7e-4 nats/token even on the highest-RA-weight tokens — there is
    nothing to distill, which is why every weighting scheme scores the same downstream. The
    teacher's own hi-vs-ctrl gap on those tokens is ~1800x larger; this target promotes that gap
    from a scalar weight to the actual distillation signal.

    Safety properties:
      - Always a valid distribution (softmax of a real vector).
      - Support ⊆ hi's plausibility set: mass can only be redistributed among tokens the hi
        teacher already considers plausible (log p_hi within log β of the max) — the student can
        never be taught a token the plain teacher wouldn't produce.
      - ``exclude_token_ids`` (e.g. <|im_end|>/<|endoftext|>): the tilt is disabled for these
        vocab entries (they keep their plain lp_hi score) — the offline pre-check caught the raw
        tilt boosting early-termination tokens, which would silently shorten responses if trained.
      - ``position_gate`` (bool, (bs, seq)): positions where the gate is False keep the plain hi
        target entirely (target = p_hi ⇒ ~zero gradient there), confining the intervention to
        positions with real visual evidence.

    Inputs are temperature-softened log-probs (as produced upstream); output is log-probs in the
    same convention, detached.
    """
    lp_hi = teacher_hi_all_log_probs
    lp_ctrl = teacher_ctrl_all_log_probs
    # anchor_coef controls how much of the plain hi log-prob anchor is kept in the tilt.
    # anchor_coef=1.0 (default) -> softmax(lp_hi + alpha*(lp_hi - lp_ctrl)) (current ours).
    # anchor_coef=0.0 + alpha=1.0 -> softmax(lp_hi - lp_ctrl) = pure contrastive-decoding target
    # (no expert anchor). The beta plausibility mask below still restricts the support to hi's
    # plausible set, which is the sole guard against ratio-driven garbage tokens when anchor=0.
    tilted = float(anchor_coef) * lp_hi + float(alpha) * (lp_hi - lp_ctrl)

    if exclude_token_ids:
        idx = torch.tensor(list(exclude_token_ids), device=lp_hi.device, dtype=torch.long)
        tilted.index_copy_(-1, idx, lp_hi.index_select(-1, idx))

    max_lp = lp_hi.max(dim=-1, keepdim=True).values
    plausible = lp_hi >= (max_lp + torch.log(torch.tensor(float(beta), device=lp_hi.device)))
    tilted = tilted.masked_fill(~plausible, float("-inf"))
    # Clamp the -inf log-probs of masked-out tokens to a large finite negative: exp(-1e4) is
    # exactly 0.0 in fp32 (numerically identical distribution) but avoids the 0 * (-inf) = NaN
    # that F.kl_div(..., log_target=True) produces on true -inf entries.
    target = torch.log_softmax(tilted, dim=-1).clamp_min(-1e4)

    if position_gate is not None:
        gate = position_gate.to(dtype=torch.bool).unsqueeze(-1)
        target = torch.where(gate, target, lp_hi)

    # Drift monitoring: KL(target || p_hi) and argmax-change rate. Must be masked to real
    # (non-padding) positions: padded rows carry all-zero lp_hi vectors (from pad_input), which
    # are not valid log-distributions and each contribute ≈ -log(vocab) to an unmasked mean —
    # observed as an impossible negative "KL" of ~-10 before this mask existed. (The training
    # loss was never affected: it applies its own loss_mask downstream.)
    kl_vs_hi = F.kl_div(lp_hi, target, reduction="none", log_target=True).sum(-1)
    argmax_changed = (target.argmax(-1) != lp_hi.argmax(-1)).float()
    if metric_mask is not None:
        m = metric_mask.to(dtype=kl_vs_hi.dtype)
        denom = m.sum().clamp(min=1.0)
        kl_mean = (kl_vs_hi * m).sum() / denom
        change_rate = (argmax_changed * m).sum() / denom
    else:
        kl_mean = kl_vs_hi.mean()
        change_rate = argmax_changed.mean()
    metrics = {
        "ra_vad/contrast_target_kl_vs_hi": kl_mean.item(),
        "ra_vad/contrast_argmax_change_rate": change_rate.item(),
    }
    return target.detach(), metrics


def ra_kd_loss(
    student_all_log_probs: torch.Tensor,
    teacher_all_log_probs: torch.Tensor,
    ra_weights: torch.Tensor,
    response_mask: torch.Tensor,
    self_distillation_mask: Optional[torch.Tensor] = None,
    *,
    logp_hi: Optional[torch.Tensor] = None,
    logp_ctrl: Optional[torch.Tensor] = None,
    temperature: float = 2.0,
    divergence_alpha: float = 0.0,
    weighting_mode: str = "continuous",
    shuffle_seed: int = 12345,
    vaopd_pv: float = 0.2,
    vaopd_lambda: float = 0.5,
    rollout_reweight: bool = False,
    rollout_tau: float = 1.0,
    uids: Optional[Sequence[Any]] = None,
    target_mode: str = "teacher",
    teacher_ctrl_all_log_probs: Optional[torch.Tensor] = None,
    contrast_alpha: float = 1.0,
    contrast_beta: float = 0.1,
    contrast_anchor_coef: float = 1.0,
    contrast_gate_positive_only: bool = False,
    contrast_exclude_token_ids: Optional[Sequence[int]] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    loss_mask = response_mask.to(dtype=student_all_log_probs.dtype)
    if self_distillation_mask is not None:
        loss_mask = loss_mask * self_distillation_mask.unsqueeze(1).to(dtype=loss_mask.dtype)
    weights = ra_weights.to(dtype=loss_mask.dtype) * loss_mask

    contrast_metrics: dict[str, Any] = {}
    if target_mode == "contrast":
        if teacher_ctrl_all_log_probs is None:
            raise ValueError("target_mode='contrast' requires teacher_ctrl_all_log_probs (full-vocab ctrl log-probs).")
        position_gate = (ra_weights > 0) if contrast_gate_positive_only else None
        teacher_all_log_probs, contrast_metrics = build_contrast_target(
            teacher_hi_all_log_probs=teacher_all_log_probs,
            teacher_ctrl_all_log_probs=teacher_ctrl_all_log_probs,
            alpha=contrast_alpha,
            beta=contrast_beta,
            anchor_coef=contrast_anchor_coef,
            position_gate=position_gate,
            exclude_token_ids=contrast_exclude_token_ids,
            metric_mask=loss_mask,
        )
    elif target_mode != "teacher":
        raise ValueError(f"Unsupported RA-VAD target_mode: {target_mode}")

    per_token_kl = token_divergence(
        teacher_all_log_probs=teacher_all_log_probs,
        student_all_log_probs=student_all_log_probs,
        alpha=divergence_alpha,
        temperature=temperature,
    )
    ra_raw = None
    rollout_sample_weights = None
    rollout_metrics: dict[str, Any] = {}
    if logp_hi is not None and logp_ctrl is not None:
        ra_raw = (logp_hi - logp_ctrl) * loss_mask
    if rollout_reweight:
        if ra_raw is None:
            raise ValueError("rollout_reweight=True requires logp_hi and logp_ctrl.")
        rollout_sample_weights, rollout_metrics = _sample_rollout_weights(
            ra_raw=ra_raw.detach(),
            loss_mask=loss_mask,
            uids=uids,
            tau=rollout_tau,
        )

    if weighting_mode == "continuous":
        if rollout_sample_weights is None:
            denom = weights.sum().clamp(min=1.0)
            loss = (per_token_kl * weights).sum() / denom
        else:
            per_sample_denom = weights.sum(dim=-1).clamp(min=1.0)
            per_sample_loss = (per_token_kl * weights).sum(dim=-1) / per_sample_denom
            active = (weights.sum(dim=-1) > 0).to(dtype=loss_mask.dtype) * rollout_sample_weights
            denom = active.sum().clamp(min=1.0)
            loss = (per_sample_loss * active).sum() / denom
    elif weighting_mode == "vaopd_grouped":
        if ra_raw is None:
            raise ValueError("weighting_mode='vaopd_grouped' requires logp_hi and logp_ctrl.")
        loss, grouped_metrics = _grouped_ra_loss(
            per_token_loss=per_token_kl,
            ra_raw=ra_raw.detach(),
            loss_mask=loss_mask,
            pv=vaopd_pv,
            group_lambda=vaopd_lambda,
            rollout_sample_weights=rollout_sample_weights,
        )
    elif weighting_mode == "shuffled_control":
        weights = _shuffle_weights_per_row(weights, loss_mask, seed=shuffle_seed)
        if rollout_sample_weights is None:
            denom = weights.sum().clamp(min=1.0)
            loss = (per_token_kl * weights).sum() / denom
        else:
            per_sample_denom = weights.sum(dim=-1).clamp(min=1.0)
            per_sample_loss = (per_token_kl * weights).sum(dim=-1) / per_sample_denom
            active = (weights.sum(dim=-1) > 0).to(dtype=loss_mask.dtype) * rollout_sample_weights
            denom = active.sum().clamp(min=1.0)
            loss = (per_sample_loss * active).sum() / denom
    else:
        raise ValueError(f"Unsupported RA-VAD weighting_mode: {weighting_mode}")

    denom = weights.sum().clamp(min=1.0)
    valid_count = loss_mask.sum().clamp(min=1.0)
    metrics = {
        "ra_vad/kd_loss": loss.detach().item(),
        "ra_vad/token_kl_mean": ((per_token_kl * loss_mask).sum() / valid_count).detach().item(),
        "ra_vad/weighted_token_kl_mean": ((per_token_kl * weights).sum() / denom).detach().item(),
        "ra_vad/num_weighted_tokens": weights.sum().detach().item(),
        "ra_vad/divergence_alpha": float(divergence_alpha),
        f"ra_vad/weighting_mode/{weighting_mode}": 1.0,
    }
    if weighting_mode == "vaopd_grouped":
        metrics.update(grouped_metrics)
    metrics.update(rollout_metrics)
    metrics.update(contrast_metrics)
    return loss, metrics
