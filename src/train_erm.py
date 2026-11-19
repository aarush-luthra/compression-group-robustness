"""Phase 1: ERM baseline. ImageNet-pretrained ResNet-18, cross-entropy,
checkpoint selected by VALIDATION worst-group accuracy. Test is touched once,
at the very end.

Usage:
    python -m src.train_erm --data_root /path/to/waterbirds --seed 0
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision

from .data import get_loaders, group_counts
from .evaluate import evaluate_and_log, predict
from .metrics import summarize


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_resnet18(pretrained=True):
    w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.resnet18(weights=w)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--results_csv", default="results/results.csv")
    a = ap.parse_args()

    set_seed(a.seed)
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", dev)
    loaders = get_loaders(a.data_root, a.batch_size, a.num_workers, a.seed)
    print("train group counts:", group_counts(loaders["train"].dataset))

    model = build_resnet18().to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(a.out_dir, exist_ok=True)
    ckpt = os.path.join(a.out_dir, f"erm_seed{a.seed}.pt")
    best = -1.0
    for ep in range(a.epochs):
        model.train()
        for x, y, _ in loaders["train"]:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        sched.step()
        v = summarize(*predict(model, loaders["val"], dev))
        print(f"ep {ep:02d} val overall={v['overall']:.4f} wga={v['wga']:.4f}")
        if v["wga"] > best:  # select on worst-group, NOT average accuracy
            best = v["wga"]
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=dev))
    name = "erm"
    evaluate_and_log(model, loaders["val"], name, a.seed, "val", a.results_csv, dev)
    r = evaluate_and_log(model, loaders["test"], name, a.seed, "test", a.results_csv, dev)
    print("TEST:", r)


if __name__ == "__main__":
    main()
