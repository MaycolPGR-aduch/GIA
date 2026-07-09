"""Generate confusion-matrix figures from already-trained GIA gesture models.

The matrix data lives inside each model's ``training_summary`` (saved by
``GestureClassifier.fit``), so no retraining is needed. New models also get this
figure automatically at train time; this script is for (re)generating figures
from existing models — e.g. for the paper.

Usage (from the repo root, using the project venv):
    python scripts/plot_confusion_matrix.py "Maycol 2"            # active version
    python scripts/plot_confusion_matrix.py "Maycol 2" --version 3
    python scripts/plot_confusion_matrix.py --all                 # every profile
    python scripts/plot_confusion_matrix.py                       # list profiles

The PNG is written next to the model in data/calibration/<profile>/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config_store import profile_calibration_dir  # noqa: E402
from app.confusion_plot import render_from_training_summary  # noqa: E402
from app.gesture_ml import GestureClassifier  # noqa: E402


def profiles_with_models() -> list[str]:
    base = ROOT / "data" / "calibration"
    names = []
    if base.exists():
        for entry in sorted(base.iterdir()):
            if entry.is_dir() and (entry / "model_registry.json").exists():
                names.append(entry.name)
    return names


def render_one(profile: str, version: int | None = None) -> Path | None:
    classifier = GestureClassifier(profile)
    if not classifier.load(version):
        print(f"[skip] {profile}: no trained model found")
        return None
    ver = classifier.active_version if classifier.active_version is not None else "legacy"
    out_path = profile_calibration_dir(profile) / f"{profile}_confusion_matrix_v{ver}.png"
    try:
        render_from_training_summary(
            classifier.training_summary,
            out_path,
            title=f"Gesture confusion matrix - {profile} (v{ver})",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] {profile} v{ver}: {exc}")
        return None
    acc = classifier.training_summary.get("validation_accuracy")
    acc_txt = f" | val acc {acc * 100:.1f}%" if acc is not None else ""
    print(f"[ok]   {out_path}{acc_txt}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render GIA gesture confusion matrices.")
    parser.add_argument("profile", nargs="?", help="Profile name (omit to list available profiles)")
    parser.add_argument("--version", type=int, default=None, help="Model version (default: active)")
    parser.add_argument("--all", action="store_true", help="Render for every profile with a model")
    args = parser.parse_args()

    if args.all:
        found = profiles_with_models()
        if not found:
            print("No profiles with trained models found.")
            return
        for name in found:
            render_one(name)
    elif args.profile:
        render_one(args.profile, args.version)
    else:
        print("Profiles with trained models:")
        for name in profiles_with_models():
            print(f"  - {name}")
        print("\nRun:  python scripts/plot_confusion_matrix.py \"<profile>\" [--version N]")


if __name__ == "__main__":
    main()
