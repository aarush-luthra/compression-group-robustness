"""Pure-numpy group metrics (no torch dependency, easy to unit test)."""
import numpy as np

N_GROUPS = 4
# group = 2*y + place ; y: 0 landbird, 1 waterbird ; place: 0 land, 1 water
GROUP_NAMES = ["land_on_land", "land_on_water", "water_on_land", "water_on_water"]


def make_group(y, place):
    return 2 * np.asarray(y) + np.asarray(place)


def group_accuracies(preds, labels, groups, n_groups=N_GROUPS):
    """Per-group accuracy (each group averaged on its own, NOT pooled)."""
    preds, labels, groups = map(np.asarray, (preds, labels, groups))
    correct = preds == labels
    accs = np.full(n_groups, np.nan)
    for g in range(n_groups):
        m = groups == g
        if m.any():
            accs[g] = correct[m].mean()
    return accs


def summarize(preds, labels, groups):
    """Overall accuracy, 4 group accuracies, worst-group accuracy."""
    accs = group_accuracies(preds, labels, groups)
    out = {"overall": float((np.asarray(preds) == np.asarray(labels)).mean())}
    for g, a in enumerate(accs):
        out[f"g{g}"] = float(a)
    out["wga"] = float(np.nanmin(accs))
    return out
