"""Render confusion-matrix figures for GIA gesture models.

The confusion matrix and class list are already computed and stored inside each
model's ``training_summary`` (see ``GestureClassifier.fit``). This module only
turns that data into a publication-quality PNG. matplotlib is imported lazily so
the runtime/training import path stays light.
"""
from __future__ import annotations

from pathlib import Path

# Human-readable, English display labels for the paper figure.
GESTURE_LABELS = {
    "neutral": "Neutral",
    "left_blink_intent": "Left wink",
    "right_blink_intent": "Right wink",
    "both_eyes_closed_intent": "Both eyes closed",
    "mouth_open_hold": "Mouth open (hold)",
    "smile": "Smile",
    "confirm": "Confirm",
    "brows_up": "Brows up",
}


def _display_labels(classes, class_labels=None):
    mapping = dict(GESTURE_LABELS)
    if class_labels:
        mapping.update(class_labels)
    return [mapping.get(str(c), str(c)) for c in classes]


def render_confusion_matrix(
    confusion,
    classes,
    out_path,
    *,
    accuracy=None,
    title=None,
    class_labels=None,
    normalize=False,
    dpi=200,
):
    """Save a confusion-matrix figure to ``out_path`` and return the Path.

    ``confusion`` is a square list-of-lists (rows = true, cols = predicted) and
    ``classes`` is the label order matching its rows/cols.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless: no display needed
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.array(confusion, dtype=float)
    if cm.size == 0 or cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("confusion matrix must be a non-empty square matrix")

    counts = cm.copy()
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    labels = _display_labels(classes, class_labels)
    n = len(labels)

    fig, ax = plt.subplots(figsize=(max(4.5, n * 0.95), max(4.0, n * 0.85)))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=(1.0 if normalize else cm.max() or 1))

    thresh = (cm.max() or 1) / 2.0
    for i in range(n):
        for j in range(n):
            text = f"{cm[i, j]:.2f}" if normalize else f"{int(counts[i, j])}"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "#1f2937",
                fontsize=10,
            )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)

    heading = title or "Gesture classification confusion matrix"
    if accuracy is not None:
        heading += f"\n(validation accuracy = {accuracy * 100:.1f}%)"
    ax.set_title(heading, fontsize=11)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_from_training_summary(training_summary, out_path, **kwargs):
    """Render directly from a model's ``training_summary`` dict."""
    confusion = training_summary.get("confusion_matrix")
    classes = training_summary.get("classes")
    if not confusion or not classes:
        raise ValueError("training_summary has no confusion_matrix/classes")
    kwargs.setdefault("accuracy", training_summary.get("validation_accuracy"))
    return render_confusion_matrix(confusion, classes, out_path, **kwargs)
