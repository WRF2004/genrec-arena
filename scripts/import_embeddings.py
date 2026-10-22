"""Import precomputed text/image embeddings aligned by string item ID.
NPZ schema: item_ids (unicode [N]), embeddings (float [N,D]); allow_pickle=False.
Use a catalog snapshot available before the training cutoff. Run BEFORE training.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument(
        "--npz",
        required=True,
        nargs="+",
        help="one or more modalities, concatenated after per-modality normalization",
    )
    a = p.parse_args()
    root = Path(a.data)
    items = json.loads((root / "items.json").read_text())
    matrices, sources = [], []
    for file in a.npz:
        with np.load(file, allow_pickle=False) as data:
            ids, embeddings = data["item_ids"], data["embeddings"]
            if (
                embeddings.ndim != 2
                or len(ids) != len(embeddings)
                or not np.isfinite(embeddings).all()
            ):
                raise ValueError("invalid embeddings shape or nonfinite values")
            lookup = {str(i): j for j, i in enumerate(ids)}
            if len(lookup) != len(ids):
                raise ValueError("duplicate item IDs")
            missing = set(items) - lookup.keys()
            if missing:
                raise ValueError(
                    f"{len(missing)} catalog items missing; explicit imputation required"
                )
            x = embeddings[[lookup[i] for i in items]].astype(np.float32)
            x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
            matrices.append(x)
        sources.append(
            {
                "filename": Path(file).name,
                "sha256": hashlib.sha256(Path(file).read_bytes()).hexdigest(),
            }
        )
    x = np.concatenate(matrices, axis=1)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    np.save(root / "features.npy", x)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["embedding_sources"] = sources
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(x.shape)


if __name__ == "__main__":
    main()
