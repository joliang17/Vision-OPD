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

from typing import Any, Optional

import torch
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    denom = mask.sum(dim=dim).clamp(min=1.0)
    return (values * mask).sum(dim=dim) / denom


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
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute stop-gradient RA token weights from paired teacher-forced log probs."""
    mask = response_mask.to(dtype=logp_hi.dtype)
    if uniform_weight:
        weights = mask.detach()
        return weights, {
            "ra_vad/ra_raw_mean": _masked_mean(logp_hi - logp_ctrl, mask).mean().detach().item(),
            "ra_vad/ra_norm_mean": 1.0,
            "ra_vad/ra_weight_mean": _masked_mean(weights, mask).mean().detach().item(),
            "ra_vad/g_sample_mean": 1.0,
            "ra_vad/positive_token_fraction": mask.float().mean().detach().item(),
        }

    ra_raw = (logp_hi - logp_ctrl) * mask
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


def token_kl(
    teacher_all_log_probs: torch.Tensor,
    student_all_log_probs: torch.Tensor,
    *,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Per-token forward KL KL(p_teacher || p_student) over the full vocabulary."""
    kl = F.kl_div(student_all_log_probs, teacher_all_log_probs, reduction="none", log_target=True).sum(dim=-1)
    return kl * (float(temperature) ** 2)


def ra_kd_loss(
    student_all_log_probs: torch.Tensor,
    teacher_all_log_probs: torch.Tensor,
    ra_weights: torch.Tensor,
    response_mask: torch.Tensor,
    self_distillation_mask: Optional[torch.Tensor] = None,
    *,
    temperature: float = 2.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    loss_mask = response_mask.to(dtype=student_all_log_probs.dtype)
    if self_distillation_mask is not None:
        loss_mask = loss_mask * self_distillation_mask.unsqueeze(1).to(dtype=loss_mask.dtype)
    weights = ra_weights.to(dtype=loss_mask.dtype) * loss_mask
    per_token_kl = token_kl(
        teacher_all_log_probs=teacher_all_log_probs,
        student_all_log_probs=student_all_log_probs,
        temperature=temperature,
    )
    denom = weights.sum().clamp(min=1.0)
    loss = (per_token_kl * weights).sum() / denom
    valid_count = loss_mask.sum().clamp(min=1.0)
    metrics = {
        "ra_vad/kd_loss": loss.detach().item(),
        "ra_vad/token_kl_mean": ((per_token_kl * loss_mask).sum() / valid_count).detach().item(),
        "ra_vad/weighted_token_kl_mean": ((per_token_kl * weights).sum() / denom).detach().item(),
        "ra_vad/num_weighted_tokens": weights.sum().detach().item(),
    }
    return loss, metrics
