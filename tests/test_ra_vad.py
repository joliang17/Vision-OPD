import torch

from verl.trainer.ppo.ra_vad import (
    attention_to_image_score,
    build_contrast_target,
    compute_ra_weights,
    ra_kd_loss,
    token_divergence,
    token_kl,
)


def _log_probs(values: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(values, dim=-1)


def test_token_kl_matches_forward_divergence():
    teacher = _log_probs(torch.tensor([[[2.0, 0.0, -1.0]]]))
    student = _log_probs(torch.tensor([[[0.0, 1.0, -1.0]]]))

    assert torch.allclose(
        token_kl(teacher, student, temperature=1.0),
        token_divergence(teacher, student, alpha=0.0, temperature=1.0),
    )


def test_ra_kd_loss_continuous_forward_is_backward_compatible():
    teacher = _log_probs(torch.tensor([[[2.0, 0.0], [0.0, 2.0]]]))
    student = _log_probs(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
    response_mask = torch.ones(1, 2)
    weights = torch.tensor([[1.0, 3.0]])

    loss, metrics = ra_kd_loss(
        student_all_log_probs=student,
        teacher_all_log_probs=teacher,
        ra_weights=weights,
        response_mask=response_mask,
        temperature=1.0,
    )
    per_token = token_kl(teacher, student, temperature=1.0)
    expected = (per_token * weights).sum() / weights.sum()

    assert torch.allclose(loss, expected)
    assert metrics["ra_vad/weighting_mode/continuous"] == 1.0


def test_vaopd_grouped_uses_top_ra_raw_tokens():
    teacher = _log_probs(torch.tensor([[[4.0, 0.0], [0.0, 4.0], [3.0, 0.0], [0.0, 3.0]]]))
    student = _log_probs(torch.tensor([[[0.0, 4.0], [4.0, 0.0], [0.0, 3.0], [3.0, 0.0]]]))
    response_mask = torch.ones(1, 4)
    logp_hi = torch.tensor([[0.9, 0.1, 0.8, -0.2]])
    logp_ctrl = torch.zeros_like(logp_hi)
    weights, _ = compute_ra_weights(logp_hi, logp_ctrl, response_mask, no_sample_gate=True)

    loss, metrics = ra_kd_loss(
        student_all_log_probs=student,
        teacher_all_log_probs=teacher,
        ra_weights=weights,
        response_mask=response_mask,
        logp_hi=logp_hi,
        logp_ctrl=logp_ctrl,
        temperature=1.0,
        weighting_mode="vaopd_grouped",
        vaopd_pv=0.5,
        vaopd_lambda=1.0,
    )
    per_token = token_kl(teacher, student, temperature=1.0)
    expected = (per_token[0, 0] + per_token[0, 2]) / 2

    assert torch.allclose(loss, expected)
    assert metrics["ra_vad/grouped_high_token_fraction"] == 0.5


def test_rollout_reweight_is_finite_with_uid_groups():
    teacher = _log_probs(torch.randn(4, 3, 5))
    student = _log_probs(torch.randn(4, 3, 5))
    response_mask = torch.ones(4, 3)
    logp_hi = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.2, 0.2, 0.2],
        ]
    )
    logp_ctrl = torch.zeros_like(logp_hi)
    weights = torch.ones(4, 3)

    loss, metrics = ra_kd_loss(
        student_all_log_probs=student,
        teacher_all_log_probs=teacher,
        ra_weights=weights,
        response_mask=response_mask,
        logp_hi=logp_hi,
        logp_ctrl=logp_ctrl,
        temperature=1.0,
        weighting_mode="continuous",
        rollout_reweight=True,
        rollout_tau=1.0,
        uids=["a", "a", "b", "b"],
    )

    assert torch.isfinite(loss)
    assert metrics["ra_vad/rollout_reweight_group_count"] == 2.0
    assert metrics["ra_vad/rollout_reweight_complete_group_fraction"] == 1.0


def test_attention_to_image_score_picks_out_image_attending_tokens():
    # seq layout: [img, img, txt, txt(prompt end) | resp0, resp1]  (seq_len=6)
    # response_positions: predicting resp0 uses query idx 3 (prompt end), predicting resp1 uses query idx 4.
    seq_len = 6
    image_token_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])  # first two positions are image tokens
    response_positions = torch.tensor([[3, 4]], dtype=torch.long)

    # One layer, one head: query 3 attends heavily to image tokens; query 4 attends heavily to text.
    attn = torch.zeros(1, 1, seq_len, seq_len)
    attn[0, 0, 3, 0] = 0.4
    attn[0, 0, 3, 1] = 0.4
    attn[0, 0, 3, 2] = 0.2
    attn[0, 0, 4, 0] = 0.05
    attn[0, 0, 4, 1] = 0.05
    attn[0, 0, 4, 3] = 0.9

    score = attention_to_image_score([attn], image_token_mask, response_positions)
    assert score.shape == (1, 2)
    assert torch.isclose(score[0, 0], torch.tensor(0.8))
    assert torch.isclose(score[0, 1], torch.tensor(0.1))
    assert score[0, 0] > score[0, 1]


def test_attention_to_image_score_averages_across_layers_and_heads():
    seq_len = 4
    image_token_mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    response_positions = torch.tensor([[2]], dtype=torch.long)

    # Two heads in layer 1 disagree; layer 2 is unanimous. Average should reflect both.
    layer1 = torch.zeros(1, 2, seq_len, seq_len)
    layer1[0, 0, 2, 0] = 1.0  # head 0: all attention on the image token
    layer1[0, 1, 2, 1] = 1.0  # head 1: all attention on a text token
    layer2 = torch.zeros(1, 1, seq_len, seq_len)
    layer2[0, 0, 2, 0] = 1.0  # single head: all attention on the image token

    score = attention_to_image_score([layer1, layer2], image_token_mask, response_positions)
    # layer1 head-mean gives 0.5 on the image token; layer2 gives 1.0; average of the two layers = 0.75
    assert torch.isclose(score[0, 0], torch.tensor(0.75))


def test_compute_ra_weights_ra_raw_override_replaces_logprob_gap():
    logp_hi = torch.zeros(1, 3)
    logp_ctrl = torch.zeros(1, 3)  # would give ra_raw = 0 everywhere without the override
    response_mask = torch.ones(1, 3)
    override = torch.tensor([[0.1, 5.0, 0.2]])

    weights_default, _ = compute_ra_weights(logp_hi, logp_ctrl, response_mask, no_sample_gate=True)
    weights_override, metrics = compute_ra_weights(
        logp_hi, logp_ctrl, response_mask, no_sample_gate=True, ra_raw_override=override
    )

    assert torch.allclose(weights_default, torch.zeros(1, 3))
    assert weights_override[0, 1] > weights_override[0, 0]
    assert weights_override[0, 1] > weights_override[0, 2]
    assert metrics["ra_vad/ra_raw_mean"] > 0.0


def test_build_contrast_target_is_valid_distribution_and_tilts_toward_visual_evidence():
    # vocab of 4; position 0: hi strongly prefers token 2 vs ctrl (visual evidence);
    # position 1: hi == ctrl (no evidence) -> target should stay == hi.
    lp_hi = torch.log_softmax(torch.tensor([[[1.0, 0.5, 2.0, -3.0], [1.0, 1.0, 0.0, -3.0]]]), dim=-1)
    lp_ctrl = torch.log_softmax(torch.tensor([[[1.0, 0.5, 0.0, -3.0], [1.0, 1.0, 0.0, -3.0]]]), dim=-1)

    target, metrics = build_contrast_target(lp_hi, lp_ctrl, alpha=1.0, beta=0.01)

    probs = target.exp()
    assert torch.allclose(probs.sum(-1), torch.ones(1, 2), atol=1e-5)
    # visual-evidence position: token 2's mass grows relative to plain hi
    assert probs[0, 0, 2] > lp_hi.exp()[0, 0, 2]
    # no-evidence position: hi == ctrl means the tilt is a no-op; every token stays within the
    # plausibility set at beta=0.01, so the target is exactly the plain hi distribution.
    assert torch.allclose(probs[0, 1], lp_hi.exp()[0, 1], atol=1e-4)
    assert "ra_vad/contrast_target_kl_vs_hi" in metrics


def test_build_contrast_target_plausibility_mask_zeroes_implausible_tokens():
    # token 3 is implausible under hi (far below beta * max) but has a huge hi/ctrl ratio —
    # without the mask the tilt would boost it; with the mask it must stay at exactly 0.
    lp_hi = torch.log_softmax(torch.tensor([[[5.0, 4.0, 3.0, -10.0]]]), dim=-1)
    lp_ctrl = torch.log_softmax(torch.tensor([[[5.0, 4.0, 3.0, -30.0]]]), dim=-1)

    target, _ = build_contrast_target(lp_hi, lp_ctrl, alpha=2.0, beta=0.1)
    assert target.exp()[0, 0, 3].item() == 0.0


def test_build_contrast_target_exclude_ids_and_position_gate():
    lp_hi = torch.log_softmax(torch.tensor([[[2.0, 1.0, 0.5, 1.5], [2.0, 1.0, 0.5, 1.5]]]), dim=-1)
    lp_ctrl = torch.log_softmax(torch.tensor([[[0.0, 1.0, 0.5, 1.5], [0.0, 1.0, 0.5, 1.5]]]), dim=-1)

    # Excluded token 0 keeps its plain hi score: its mass must NOT be boosted by its large
    # hi/ctrl ratio the way the unexcluded run boosts it.
    boosted, _ = build_contrast_target(lp_hi, lp_ctrl, alpha=1.0, beta=0.01)
    protected, _ = build_contrast_target(lp_hi, lp_ctrl, alpha=1.0, beta=0.01, exclude_token_ids=[0])
    assert protected.exp()[0, 0, 0] < boosted.exp()[0, 0, 0]

    # Position gate: gated-off position falls back to the plain hi distribution exactly.
    gate = torch.tensor([[True, False]])
    gated, _ = build_contrast_target(lp_hi, lp_ctrl, alpha=1.0, beta=0.01, position_gate=gate)
    assert torch.allclose(gated[0, 1], lp_hi[0, 1], atol=1e-6)
    assert not torch.allclose(gated[0, 0], lp_hi[0, 0], atol=1e-4)


def test_ra_kd_loss_contrast_mode_runs_and_reports_metrics():
    torch.manual_seed(0)
    student = torch.log_softmax(torch.randn(2, 3, 8), dim=-1)
    teacher_hi = torch.log_softmax(torch.randn(2, 3, 8), dim=-1)
    teacher_ctrl = torch.log_softmax(torch.randn(2, 3, 8), dim=-1)
    response_mask = torch.ones(2, 3)
    weights = torch.rand(2, 3)

    loss, metrics = ra_kd_loss(
        student_all_log_probs=student,
        teacher_all_log_probs=teacher_hi,
        ra_weights=weights,
        response_mask=response_mask,
        temperature=1.0,
        target_mode="contrast",
        teacher_ctrl_all_log_probs=teacher_ctrl,
        contrast_alpha=0.5,
        contrast_beta=0.05,
        contrast_gate_positive_only=True,
        contrast_exclude_token_ids=[7],
    )
    assert torch.isfinite(loss)
    assert "ra_vad/contrast_target_kl_vs_hi" in metrics
    assert "ra_vad/contrast_argmax_change_rate" in metrics


def test_build_contrast_target_metric_mask_excludes_padding_rows():
    # Real position: hi/ctrl differ; padding position: all-zero "log-prob" vectors as produced
    # by pad_input — not valid distributions, must not pollute the drift metrics.
    lp_hi_real = torch.log_softmax(torch.tensor([[2.0, 1.0, 0.0, -1.0]]), dim=-1)
    lp_ctrl_real = torch.log_softmax(torch.tensor([[0.0, 1.0, 0.0, -1.0]]), dim=-1)
    lp_hi = torch.stack([lp_hi_real[0], torch.zeros(4)]).unsqueeze(0)
    lp_ctrl = torch.stack([lp_ctrl_real[0], torch.zeros(4)]).unsqueeze(0)

    _, unmasked = build_contrast_target(lp_hi, lp_ctrl, alpha=1.0, beta=0.01)
    _, masked = build_contrast_target(
        lp_hi, lp_ctrl, alpha=1.0, beta=0.01, metric_mask=torch.tensor([[1.0, 0.0]])
    )
    # unmasked mean is dragged toward -log(vocab) by the padding row; masked must be >= 0
    assert unmasked["ra_vad/contrast_target_kl_vs_hi"] < 0
    assert masked["ra_vad/contrast_target_kl_vs_hi"] >= 0
