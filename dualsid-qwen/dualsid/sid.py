"""Residual K-means tokenization of content and training-only co-user features."""
from collections import Counter, defaultdict
import math
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from .io import fingerprint


def residual_quantize(features, levels, size, seed):
    if levels < 1 or size < 2:
        raise ValueError("levels >= 1 and size >= 2 required")
    residual = np.asarray(features, dtype=np.float32).copy()
    codes, books = [], []
    for level in range(levels):
        k = min(size, len(features), len(np.unique(residual, axis=0)))
        km = KMeans(n_clusters=max(1, k), n_init=5, random_state=seed + level)
        labels = km.fit_predict(residual)
        books.append(km.cluster_centers_.tolist())
        codes.append(labels)
        residual -= km.cluster_centers_[labels]
    return np.stack(codes, axis=1), books


def addresses(codes, stream, base):
    """Append fixed-width base-K collision ranks, not a full catalog-sized vocabulary."""
    buckets = defaultdict(list)
    for i, code in enumerate(codes):
        buckets[tuple(int(c) for c in code)].append(i)
    max_bucket = max(map(len, buckets.values()))
    width = max(1, math.ceil(math.log(max_bucket, base)))
    while base ** width < max_bucket:
        width += 1
    paths = [None] * len(codes)
    for code, indices in buckets.items():
        for rank, i in enumerate(indices):
            digits = [(rank // base ** p) % base for p in reversed(range(width))]
            values = list(code) + digits
            paths[i] = [f"<{stream}{level}_{v}>" for level, v in enumerate(values)]
    usage = []
    for level in range(codes.shape[1]):
        counts = np.array(list(Counter(codes[:, level]).values()), dtype=float)
        p = counts / counts.sum()
        usage.append({"used": len(counts), "perplexity": float(np.exp(-(p * np.log(p)).sum()))})
    return paths, {"raw_collision_rate": 1 - len(buckets) / len(codes),
                   "max_bucket": max_bucket, "suffix_width": width,
                   "path_length": len(paths[0]), "layers": usage}


def fit(data, levels=3, size=64, dim=64, seed=42, embeddings=None):
    catalog = data["catalog"]
    # Frozen catalog only; metadata may have been known at train cutoff.
    if embeddings is None:
        x = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=20000).fit_transform(data["texts"])
        d = min(dim, x.shape[0] - 1, x.shape[1] - 1)
        content = normalize(TruncatedSVD(n_components=max(1, d), random_state=seed).fit_transform(x))
        content_source = "char TF-IDF + SVD"
    else:
        content = np.asarray(embeddings, dtype=np.float32)
        if content.ndim != 2 or len(content) != len(catalog) or not np.isfinite(content).all():
            raise ValueError("Embedding rows must match catalog order; all values finite")
        content = normalize(content)
        content_source = "external embeddings in catalog order"
    lookup = {item: i for i, item in enumerate(catalog)}
    rr, cc = [], []
    for user, seq in enumerate(data["train_sequences"]):
        for item in set(seq):
            rr.append(lookup[item]); cc.append(user)
    matrix = csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(len(catalog), len(data["train_sequences"])))
    if matrix.shape[1] < 2:
        raise ValueError("Need >=2 users for collaborative SVD")
    d = min(dim, min(matrix.shape) - 1)
    collaborative = normalize(TruncatedSVD(n_components=max(1, d), random_state=seed).fit_transform(matrix))
    artifact = {"catalog": catalog, "streams": {}, "diagnostics": {}, "codebooks": {},
                "config": dict(levels=levels, size=size, dim=dim, seed=seed, content=content_source),
                "data_hash": fingerprint(data)}
    for stream, features in (("s", content), ("c", collaborative)):
        codes, books = residual_quantize(features, levels, size, seed)
        paths, diagnostics = addresses(codes, stream, size)
        artifact["streams"][stream] = dict(zip(catalog, paths))
        artifact["diagnostics"][stream] = diagnostics
        artifact["codebooks"][stream] = books
    artifact["tokens"] = sorted({t for paths in artifact["streams"].values() for path in paths.values() for t in path})
    return artifact


class Trie:
    def __init__(self, paths):
        self.root = {}
        self.items = {}
        for item, path in paths.items():
            node = self.root
            key = tuple(path)
            if key in self.items:
                raise ValueError("Non-unique SID")
            self.items[key] = item
            for token in path:
                node = node.setdefault(token, {})

    def allowed(self, prefix):
        node = self.root
        for token in prefix:
            node = node.get(token, {})
        return sorted(node)
