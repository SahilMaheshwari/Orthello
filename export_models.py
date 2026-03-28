"""
export_models.py — Export trained GA checkpoints to JSON for the web frontend.

The frontend (othello_frontend.html) can load these JSON files to play against
any saved generation directly in the browser — no Python or PyTorch required.

Usage
-----
    # Export every checkpoint in ga_models/ (default)
    python export_models.py

    # Specify a different model directory
    python export_models.py --dir my_models

    # Export a single checkpoint
    python export_models.py --model ga_models/best_gen_010.pt

    # Export to a custom output folder
    python export_models.py --out-dir frontend_models

Output
------
    ga_models/best_gen_001.json   ← weights + metadata
    ga_models/best_gen_002.json
    ...
    ga_models/best_ever.json
    ga_models/manifest.json       ← index of all exported models (sorted by gen)

Requirements
------------
    pip install torch
    train_ga.py must be in the same directory (for the OthelloNet class).
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import torch

from cli_utils import get_formatter

# ── Import OthelloNet from train_ga ───────────────────────────────────
try:
    from train_ga import OthelloNet
except ImportError:
    # Fallback: define the same architecture inline so this script
    # can run even if train_ga.py has missing dependencies.
    import torch.nn as nn

    class OthelloNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(65, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 32),  nn.ReLU(),
                nn.Linear(32, 64),
            )

        def forward(self, x):
            return self.net(x)


# ── Helpers ────────────────────────────────────────────────────────────

def export_one(pt_path: str, json_path: str) -> dict:
    """Load a .pt checkpoint and write a compact JSON file."""
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)

    net = OthelloNet()
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    # Collect all named parameters as Python lists (JSON-serialisable)
    weights = {
        name: param.data.tolist()
        for name, param in net.named_parameters()
    }

    stem = Path(pt_path).stem   # e.g. "best_gen_010" or "best_ever"
    is_best_ever = stem == "best_ever"

    payload = {
        "label":       stem,
        "generation":  int(ckpt.get("generation", 0)),
        "fitness":     float(ckpt.get("fitness", 0.0)),
        "best_ever":   is_best_ever,
        # Architecture description (for the JS decoder)
        "arch": {
            "input_size":  65,
            "hidden":      [128, 64, 32],
            "output_size": 64,
            "activation":  "relu",
            # Maps parameter names to layer indices in the Sequential
            "layer_indices": [0, 2, 4, 6],
        },
        "weights": weights,
    }

    # Write compact JSON (no extra whitespace — smaller files)
    with open(json_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_kb = os.path.getsize(json_path) / 1024
    return {
        "file":       Path(json_path).name,
        "label":      stem,
        "generation": payload["generation"],
        "fitness":    payload["fitness"],
        "best_ever":  is_best_ever,
        "size_kb":    round(size_kb, 1),
    }


# ── Main ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Export Othello GA model checkpoints to JSON"
    )
    ap.add_argument("--dir",     default="ga_models",
                    help="Directory containing .pt checkpoints (default: ga_models)")
    ap.add_argument("--model",   default=None,
                    help="Export a single .pt file instead of a whole directory")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory for JSON files (default: same as --dir)")
    args = ap.parse_args()

    fmt = get_formatter()

    # Collect input files
    if args.model:
        if not os.path.isfile(args.model):
            fmt.error(f"File not found: {args.model}")
            sys.exit(1)
        pt_files = [args.model]
        out_dir = args.out_dir or os.path.dirname(args.model) or "."
    else:
        if not os.path.isdir(args.dir):
            fmt.error(f"Directory not found: {args.dir}")
            fmt.info("Train a model first with:  python train_ga.py")
            sys.exit(1)
        # Sorted generation checkpoints
        pt_files = sorted(glob.glob(os.path.join(args.dir, "best_gen_*.pt")))
        # Append best_ever at the end if it exists
        be = os.path.join(args.dir, "best_ever.pt")
        if os.path.exists(be):
            pt_files.append(be)
        out_dir = args.out_dir or args.dir

    os.makedirs(out_dir, exist_ok=True)

    if not pt_files:
        fmt.warning("No .pt files found. Train a model first:  python train_ga.py")
        sys.exit(0)

    fmt.header("💾 Othello GA Model Exporter", width=58)
    fmt.highlight(f"Exporting", f"{len(pt_files)} checkpoint(s)", color="cyan")
    fmt.highlight(f"Output directory", out_dir, color="yellow")
    print()

    manifest = []
    t0 = time.time()

    for pt_path in pt_files:
        stem = Path(pt_path).stem
        json_path = os.path.join(out_dir, stem + ".json")
        try:
            info = export_one(pt_path, json_path)
            manifest.append(info)
            tag = "⭐ best-ever" if info["best_ever"] else ""
            fmt.success(f"Gen {info['generation']:03d}  |  fitness={info['fitness']:.1f}  |  {info['size_kb']:.0f} KB {tag}")
        except Exception as exc:
            fmt.error(f"{pt_path}: {exc}")

    # Write manifest (sorted by generation, best_ever last)
    manifest.sort(key=lambda x: (x["best_ever"], x["generation"]))
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t0
    print()
    fmt.info(f"Manifest written → {manifest_path}")
    fmt.highlight("Done in", f"{elapsed:.1f}s", color="green")
    print()
    fmt.subheader("Next step")
    print("  Open othello_frontend.html and use 'Load Generations'")
    print("  to import these JSON files.\n")


if __name__ == "__main__":
    main()