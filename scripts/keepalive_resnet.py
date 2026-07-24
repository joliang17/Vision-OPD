#!/usr/bin/env python3
"""

Usage:
    python run_resnet.py                       # all 8 GPUs, save every 500 steps
    python run_resnet.py --gpus 0,1,2,3        # subset
    python run_resnet.py --batch-size 256      # raise if util < 50%; lower if OOM
    CUDA_VISIBLE_DEVICES=0,1 python run_resnet.py --gpus 0,1

Stop with Ctrl-C.
"""
import argparse
import os
import tempfile
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp


# ----------------------------- model (CIFAR ResNet18) -----------------------------
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)
            )

    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        out = out + self.short(x)
        return F.relu(out)


class ResNet18(nn.Module):
    """ResNet18 adapted for 32x32 CIFAR (3x3 stride-1 stem, no maxpool)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        self.layer1 = self._make(64, 64, 2, 1)
        self.layer2 = self._make(64, 128, 2, 2)
        self.layer3 = self._make(128, 256, 2, 2)
        self.layer4 = self._make(256, 512, 2, 2)
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _make(cin, cout, n, stride):
        layers = [BasicBlock(cin, cout, stride)]
        for _ in range(n - 1):
            layers.append(BasicBlock(cout, cout, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


# ----------------------------- data -----------------------------
def _from_hf(root):
    """HuggingFace uoft-cs/cifar10 (fast here) -> uint8 [N,3,32,32], long [N]."""
    import numpy as np
    from datasets import load_dataset

    ds = load_dataset("uoft-cs/cifar10", split="train", cache_dir=root)
    imgs = np.stack([np.asarray(im) for im in ds["img"]])  # [N,32,32,3] uint8
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(ds["label"], dtype=torch.long)
    return x, y


def _from_torchvision(root):
    from torchvision import datasets

    ds = datasets.CIFAR10(root, train=True, download=True)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
    y = torch.tensor(ds.targets, dtype=torch.long)
    return x, y


def load_cifar10_tensors(root):
    """Return (images uint8 [N,3,32,32], labels long [N], source_tag).
    """
    os.makedirs(root, exist_ok=True)
    cache = os.path.join(root, "cifar10_uint8.pt")
    if os.path.exists(cache):
        blob = torch.load(cache)
        return blob["x"], blob["y"], "cifar10(cache)"

    for name, fn in (("cifar10(hf)", _from_hf), ("cifar10(torchvision)", _from_torchvision)):
        try:
            x, y = fn(root)
            torch.save({"x": x, "y": y}, cache)  # persist for the other workers / reruns
            return x, y, name
        except Exception as e:  # noqa: BLE001
            print(f"[data] {name} failed ({type(e).__name__}: {e})")

    print("[data] falling back to synthetic random data")
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 256, (50000, 3, 32, 32), dtype=torch.uint8, generator=g)
    y = torch.randint(0, 10, (50000,), dtype=torch.long, generator=g)
    return x, y, "synthetic"


# ----------------------------- checkpoint -----------------------------
def save_latest(model, opt, step, ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, "latest.pt")
    tmp = path + ".tmp"
    torch.save({"step": step, "model": model.state_dict(), "opt": opt.state_dict()}, tmp)
    os.replace(tmp, path)  # atomic: readers never see a half-written file
    return path


# ----------------------------- per-GPU worker -----------------------------
def worker(rank, args):
    torch.set_num_threads(2)  # 8 procs share the box; don't oversubscribe CPU
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    
    torch.backends.cudnn.benchmark = True

    x_cpu, y_cpu, src = load_cifar10_tensors(args.data_root)
    x = x_cpu.to(dev)  # uint8, ~150 MB
    y = y_cpu.to(dev)
    n = x.shape[0]
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=dev).view(1, 3, 1, 1)

    model = ResNet18().to(dev)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    gen = torch.Generator(device=dev).manual_seed(1234 + rank)

    if rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        print(f"[exp] data={src} gpus={args.num_gpus} batch={args.batch_size} "
              f"save_every={args.save_every} -> {os.path.join(args.ckpt_dir, 'latest.pt')}")

    step = 0
    t0 = time.time()
    while True:
        idx = torch.randint(0, n, (args.batch_size,), device=dev, generator=gen)
        xb = x[idx].float().div_(255.0).sub_(mean).div_(std)
        yb = y[idx]
        out = model(xb)
        loss = F.cross_entropy(out, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1

        if rank == 0 and step % args.save_every == 0:
            save_latest(model, opt, step, args.ckpt_dir)

        if rank == 0 and step % args.log_every == 0:
            torch.cuda.synchronize(dev)
            dt = time.time() - t0
            t0 = time.time()
            ips = args.log_every * args.batch_size / dt
            mem = torch.cuda.max_memory_allocated(dev) / 1024**2
            print(f"[step {step:>7}] loss {loss.item():.3f} | {ips:6.0f} img/s/gpu | "
                  f"peak mem {mem:.0f} MB/gpu")


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7",
                    help="select GPUs to use")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="per-GPU batch size")
    ap.add_argument("--save-every", type=int, default=500, help="steps between latest.pt saves")
    ap.add_argument("--log-every", type=int, default=100, help="steps between throughput logs")
    ap.add_argument("--ckpt-dir", default=os.path.join(tempfile.gettempdir(), "save"),
                    help="dir for latest.pt (default: <tmp>/save)")
    ap.add_argument("--data-root", default=os.path.join(tempfile.gettempdir(), "cifar10"),
                    help="CIFAR-10 download/cache dir")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args.num_gpus = len(args.gpus.split(","))

    load_cifar10_tensors(args.data_root)

    try:
        mp.spawn(worker, args=(args,), nprocs=args.num_gpus, join=True)
    except KeyboardInterrupt:
        print("\n[exp] stopped.")


if __name__ == "__main__":
    main()
