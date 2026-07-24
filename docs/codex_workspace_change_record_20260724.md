# Codex workspace change record — 2026-07-24

This document records workspace mutations made by Codex while repairing the
`ra-vad` Git history. It exists so the changes can be audited or reversed
without guessing.

## Scope

Within `Vision-OPD`, Codex changed working files in only these categories:

1. `.gitignore`
2. Shell-script organization and references to moved shell scripts
3. This audit document and `experiment_script/README.md`

Codex did not intentionally edit the RA-VAD algorithm, `verl/` Python
implementation, checkpoints, model weights, evaluation records, benchmark
images, rollouts, or other experiment data.

## Shell-script organization

### Moves

- 303 one-off `scripts/*.sh` launchers were moved to
  `experiment_script/<same-basename>.sh`.
- `scripts/mlx_batch_20260718/` was moved to
  `experiment_script/mlx_batch_20260718/`.
- `eval/logs/run_model_eval.sh` was moved to
  `experiment_script/run_model_eval.sh`.
- Reusable entry points and helpers remain under `scripts/`, including:
  `run_vision_opd*.sh`, `run_experiment_*.sh`, `merge_checkpoint.sh`,
  `prune_checkpoints.sh`, `run_zoombench_canonical.sh`, `rejudge_one.sh`,
  `_zoom_eval_generic.sh`, and `setup_qwen35_env.sh`.
- `experiment_script/README.md` was added to document this policy.

The basename was preserved for every moved script. Therefore, except for
`run_model_eval.sh`, the inverse mapping is mechanically:

```text
experiment_script/<name>.sh -> scripts/<name>.sh
experiment_script/mlx_batch_20260718/ -> scripts/mlx_batch_20260718/
```

For the special case:

```text
experiment_script/run_model_eval.sh -> eval/logs/run_model_eval.sh
```

### Reference rewrites

Text references were mechanically rewritten:

```text
scripts/<moved-name>.sh -> experiment_script/<moved-name>.sh
eval/logs/run_model_eval.sh -> experiment_script/run_model_eval.sh
```

The rewrite was applied to matching shell, Python, Markdown, HTML, YAML,
TOML, and text files. Reversal should use the same basename mapping instead
of a broad replacement of every `experiment_script/` occurrence.

All 338 shell scripts under `scripts/`, `experiment_script/`, and `eval/`
passed `bash -n` after organization.

## `.gitignore`

Before Codex edits, `.gitignore` contained only:

```gitignore
__pycache__/
*.py[cod]
*.log

checkpoints/
data/
external/
logs/
outputs/
rollouts/
tensorboard_log/
vlmevalkit_outputs/

wandb/
```

Codex then:

- anchored repository-level data directories with leading `/`;
- ignored raster images and PDFs;
- ignored local Python compatibility shims and Claude session state;
- ignored analysis, evaluation, figure, judge, and cache outputs;
- ignored downloaded benchmark data and evaluation answer directories;
- ignored generated VDH/FCE artifacts;
- ignored machine-specific MLX YAML files;
- explicitly kept nested source directories such as
  `verl/trainer/config/data/` trackable.

The current `.gitignore` is authoritative for the exact final rule set.

## Git preservation points

- Original orphan commit history:
  `backup/ra-vad-orphan-20260724` at `ed5f463`
- Local Git bundle:
  `../git-backups/Vision-OPD-ra-vad-before-repair-20260724.bundle`
- Complete non-data orphan snapshot:
  `backup/ra-vad-orphan-full-snapshot-20260724` at `ab5f93a`
- Normal-history recovery:
  `recovery/ra-vad-code-20260724` at `25946f9`
- Repaired remote branch:
  `origin/ra-vad` at `25946f9`

At verification time:

- the complete orphan snapshot tracked 875 non-data files;
- every tracked snapshot file matched the current workspace by Git blob hash;
- mismatch count was zero;
- non-ignored files missing from the snapshot were zero;
- the complete orphan snapshot tree and normal-history recovery tree were
  byte-for-byte identical.

## Worktree roles

- `Vision-OPD` preserves the original experiment directory and its ignored
  data. Its branch was aligned to the complete orphan snapshot so changes to
  every non-data file are visible through `git status`.
- `Vision-OPD-repair-validation` is the clean normal-history `ra-vad`
  worktree tracking `origin/ra-vad`.

No `git clean`, hard reset, or checkout that updates the original experiment
directory's working files was used. The final alignment used a mixed reset,
which updated only the branch/index and left working files in place.

## External cleanup performed by explicit user request

Outside `Vision-OPD`, Codex also removed:

- `Zooming-without-Zooming`
- `simple-mmeval`
- the `qgen_opsd` auxiliary worktree after merging it
- three auxiliary `unsup-opsd` worktrees

These deletions are separate from the `Vision-OPD` history repair.
