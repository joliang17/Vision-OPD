#!/usr/bin/env python3
"""
Visual-dependency per-token highlight for the CS-OPSD paper (Fig. "token delta").

For each sample (one fixed, teacher-forced response) and each teacher checkpoint
(base / OPSD / ours), compute per response token
    delta_t = log p(y_t | y_<t, real_image) - log p(y_t | y_<t, control_image)
control_image = the method's content-erased control (default: black).
delta_t > 0  =>  that token depends on actually seeing the image.

Output: one JSON per sample: {tokens:[...], deltas:{base:[...],opsd:[...],ours:[...]}}.
The paper-side (no-GPU) renderer turns these into Fig.5-style highlighted text.
THIS SCRIPT NEEDS A GPU.

BATCH run over a manifest (recommended -- loads each teacher only once):
    cd /mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/Vision-OPD
    source .../miniconda3/etc/profile.d/conda.sh && conda activate qwen35
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
    python3 scripts/visual_dependency_highlight.py \
        --manifest docs/vdh_manifest.jsonl \
        --base <Qwen3.5-2B base dir> \
        --opsd <MERGED answer-hint OPSD ckpt dir> \
        --ours <MERGED CS-OPSD ckpt dir> \
        --control black \
        --out-dir docs/vdh_out

manifest = one JSON object per line, with keys:
    id, image, prompt_file, response_file   (paths relative to Vision-OPD/)

Notes the GPU box must verify:
  * MODEL CLASS: Qwen3-VL / Qwen3.5-VL load via AutoModelForImageTextToText in
    recent transformers; if it fails, use the model's own class. processor=AutoProcessor.
  * base/opsd/ours must be the SAME size (all Qwen3.5-2B) so the shared response
    tokenizes identically -> tokens align 1:1 across teachers.
  * Merge FSDP checkpoints first (scripts/merge_checkpoint.sh) so --opsd/--ours
    point at a HF-loadable dir (config.json present).
"""
import argparse, json, os
import torch
from PIL import Image


def make_control(image, kind):
    if kind == "black":
        return Image.new("RGB", image.size, (0, 0, 0))
    if kind == "gray":
        return Image.new("RGB", image.size, (127, 127, 127))
    if kind == "gaussian":
        arr = (torch.randn(image.size[1], image.size[0], 3).numpy() * 64 + 127).clip(0, 255).astype("uint8")
        return Image.fromarray(arr)
    raise ValueError(kind)


def build_inputs(processor, image, prompt, response, device):
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": response}]},
    ]
    full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prefix = processor.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[full], images=[image], return_tensors="pt").to(device)
    prefix_ids = processor(text=[prefix], images=[image], return_tensors="pt").input_ids[0]
    resp_start = int(prefix_ids.shape[0])
    return inputs, resp_start


@torch.no_grad()
def token_logps(model, inputs, resp_start):
    logits = model(**inputs).logits[0].float()
    logp = torch.log_softmax(logits, dim=-1)
    ids = inputs["input_ids"][0]
    next_logp = logp[:-1].gather(1, ids[1:, None]).squeeze(1)
    ids_resp = ids[resp_start:]
    logp_resp = next_logp[resp_start - 1:]
    n = min(len(ids_resp), len(logp_resp))
    return ids_resp[:n], logp_resp[:n]


def load_teacher(path, processor_path, device):
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as _Model
    except Exception:
        from transformers import AutoModelForVision2Seq as _Model
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    model = _Model.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
    return model, processor


def score_sample(model, processor, image, control, prompt, response, device):
    inp_i, rs_i = build_inputs(processor, image, prompt, response, device)
    ids_i, lp_i = token_logps(model, inp_i, rs_i)
    inp_c, rs_c = build_inputs(processor, control, prompt, response, device)
    ids_c, lp_c = token_logps(model, inp_c, rs_c)
    n = min(len(ids_i), len(ids_c))
    toks = [processor.tokenizer.decode([int(t)]) for t in ids_i[:n]]
    delta = (lp_i[:n] - lp_c[:n]).tolist()
    return toks, delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--opsd", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--processor", default=None, help="processor dir; default = --base")
    ap.add_argument("--control", default="black", choices=["black", "gray", "gaussian"])
    # batch
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out-dir", default=None)
    # single-sample fallback
    ap.add_argument("--image"); ap.add_argument("--prompt-file")
    ap.add_argument("--response-file"); ap.add_argument("--out")
    args = ap.parse_args()

    device = "cuda"
    proc = args.processor or args.base

    if args.manifest:
        samples = [json.loads(l) for l in open(args.manifest) if l.strip()]
    else:
        samples = [dict(id=os.path.splitext(os.path.basename(args.out))[0],
                        image=args.image, prompt_file=args.prompt_file, response_file=args.response_file)]
    out_dir = args.out_dir or (os.path.dirname(args.out) if args.out else "vdh_out")
    os.makedirs(out_dir, exist_ok=True)

    # results[id] = {"tokens":..., "deltas":{teacher:...}, "toks_by":{teacher:...}}
    results = {s["id"]: {"tokens": None, "deltas": {}, "toks_by": {}} for s in samples}

    for name, path in [("base", args.base), ("opsd", args.opsd), ("ours", args.ours)]:
        print(f"[VDH] loading teacher = {name} ({path})")
        model, processor = load_teacher(path, proc, device)
        for s in samples:
            image = Image.open(s["image"]).convert("RGB")
            control = make_control(image, args.control)
            prompt = open(s["prompt_file"]).read().strip()
            response = open(s["response_file"]).read().strip()
            toks, delta = score_sample(model, processor, image, control, prompt, response, device)
            results[s["id"]]["deltas"][name] = delta
            results[s["id"]]["toks_by"][name] = toks
            print(f"    {s['id']}: {len(toks)} resp tokens")
        del model
        torch.cuda.empty_cache()

    for sid, r in results.items():
        # use ours tokens as canonical; verify alignment
        ref = r["toks_by"].get("ours") or next(iter(r["toks_by"].values()))
        r["tokens"] = ref
        aligned = all(r["toks_by"][t] == ref for t in r["toks_by"])
        out = {"id": sid, "control": args.control, "aligned": aligned,
               "tokens": r["tokens"], "deltas": r["deltas"]}
        if not aligned:
            out["toks_by"] = r["toks_by"]
            print(f"  WARNING {sid}: teacher token mismatch; kept per-teacher tokens")
        json.dump(out, open(os.path.join(out_dir, f"{sid}.json"), "w"), ensure_ascii=False, indent=1)
    print(f"[VDH] wrote {len(results)} json files -> {out_dir}")


if __name__ == "__main__":
    main()
