"""Shared evaluation harness. Every phase (ERM, pruned, quantized, distilled)
must go through `evaluate_and_log` so comparisons use one code path."""
import io
import os
import time

import numpy as np
import pandas as pd
import torch

from .metrics import summarize

CSV_COLUMNS = ["model", "seed", "split", "overall", "g0", "g1", "g2", "g3",
               "wga", "size_mb", "latency_ms"]


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    P, Y, G = [], [], []
    for x, y, g in loader:
        P.append(model(x.to(device)).argmax(1).cpu())
        Y.append(y)
        G.append(g)
    return (torch.cat(P).numpy(), torch.cat(Y).numpy(), torch.cat(G).numpy())


def model_size_mb(model):
    """Serialized state_dict size in MB (works for pruned and quantized models)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


@torch.no_grad()
def latency_ms(model, device="cpu", batch_size=1, warmup=10, iters=50, threads=1):
    """Mean per-batch latency. Use the SAME device/threads for every model.
    INT8 (fbgemm/qnnpack) only runs on CPU, so CPU is the fair default."""
    torch.set_num_threads(threads)
    model.eval()
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    for _ in range(warmup):
        model(x)
    if device != "cpu":
        torch.cuda.synchronize()
    t = []
    for _ in range(iters):
        s = time.perf_counter()
        model(x)
        if device != "cpu":
            torch.cuda.synchronize()
        t.append((time.perf_counter() - s) * 1e3)
    return float(np.mean(t))


def evaluate_and_log(model, loader, name, seed, split, csv_path,
                     device="cpu", lat_device="cpu", measure_latency=True):
    """Evaluate `model` on `loader` and append one row to csv_path.
    NOTE: only call with split='test' for final evaluation."""
    preds, labels, groups = predict(model, loader, device)
    row = {"model": name, "seed": seed, "split": split, **summarize(preds, labels, groups)}
    row["size_mb"] = model_size_mb(model)
    if measure_latency:
        cpu_model = model if lat_device == "cpu" and device == "cpu" else None
        if cpu_model is None and lat_device == "cpu":
            import copy
            cpu_model = copy.deepcopy(model).cpu()
        row["latency_ms"] = latency_ms(cpu_model or model, lat_device)
    else:
        row["latency_ms"] = float("nan")
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df = pd.DataFrame([row], columns=CSV_COLUMNS)
    df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
    return row
