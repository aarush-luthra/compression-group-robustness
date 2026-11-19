import numpy as np
from src.metrics import group_accuracies, make_group, summarize


def test_make_group():
    assert list(make_group([0, 0, 1, 1], [0, 1, 0, 1])) == [0, 1, 2, 3]


def test_group_accuracy_not_pooled():
    # group 0: 100 samples all correct; group 3: 2 samples, 1 correct.
    groups = np.array([0] * 100 + [3] * 2 + [1, 2])
    labels = np.zeros(104, dtype=int)
    preds = labels.copy()
    preds[101] = 1  # one wrong in group 3
    preds[102] = 1  # group 1 wrong
    preds[103] = 1  # group 2 wrong
    accs = group_accuracies(preds, labels, groups)
    assert accs[0] == 1.0 and accs[3] == 0.5 and accs[1] == 0.0
    s = summarize(preds, labels, groups)
    assert s["wga"] == 0.0
    assert s["overall"] > 0.95  # average hides the worst group
