"""Residual quantization plus collision-free radix suffix and trie decoding."""

import math
from collections import Counter, defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def residual_kmeans(x, levels, size, seed=42, iterations=30):
    rng = np.random.default_rng(seed)
    residual = x.astype(np.float32).copy()
    books, codes = [], []
    for _ in range(levels):
        centers = residual[rng.choice(len(x), size, replace=len(x) < size)].copy()
        for _ in range(iterations):
            labels = ((residual[:, None] - centers[None]) ** 2).sum(-1).argmin(-1)
            updated = centers.copy()
            for j in range(size):
                members = residual[labels == j]
                updated[j] = (
                    members.mean(0) if len(members) else residual[rng.integers(len(x))]
                )
            if np.allclose(updated, centers):
                break
            centers = updated
        labels = ((residual[:, None] - centers[None]) ** 2).sum(-1).argmin(-1)
        residual -= centers[labels]
        books.append(centers)
        codes.append(labels)
    return np.stack(codes, 1), np.stack(books), float(np.mean(residual**2))


class RQVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, levels, size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, input_dim),
        )
        self.books = nn.Parameter(torch.randn(levels, size, latent_dim) * 0.02)

    def forward(self, x):
        z = self.encoder(x)
        residual, quantized = z, torch.zeros_like(z)
        loss, codes = z.new_zeros(()), []
        for book in self.books:
            index = torch.cdist(residual, book).argmin(-1)
            q = book[index]
            loss = (
                loss
                + F.mse_loss(q, residual.detach())
                + 0.25 * F.mse_loss(residual, q.detach())
            )
            quantized = quantized + q
            residual = residual - q.detach()
            codes.append(index)
        straight = z + (quantized - z).detach()
        reconstruction = F.mse_loss(self.decoder(straight), x)
        return reconstruction + loss, torch.stack(codes, -1), reconstruction


def fit_rqvae(x, levels, size, seed=42, epochs=30, device="cpu"):
    torch.manual_seed(seed)
    model = RQVAE(x.shape[1], min(32, x.shape[1]), levels, size).to(device)
    data = torch.as_tensor(x, device=device)
    # K-means initialization avoids starting every level in the same tiny random region.
    with torch.no_grad():
        z = model.encoder(data).cpu().numpy()
        _, books, _ = residual_kmeans(z, levels, size, seed)
        model.books.copy_(torch.tensor(books, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        for idx in torch.randperm(len(data), device=device).split(256):
            loss, _, _ = model(data[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        _, codes, recon = model(data)
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    return codes.cpu().numpy(), state, float(recon)


class SemanticIndex:
    PAD, BOS, SEP = 0, 1, 2

    def __init__(self, raw_codes, size):
        raw_codes = np.asarray(raw_codes, dtype=int)
        if size < 2 or raw_codes.ndim != 2 or len(raw_codes) == 0:
            raise ValueError("nonempty matrix and codebook size >=2 required")
        if raw_codes.min() < 0 or raw_codes.max() >= size:
            raise ValueError("code outside codebook")
        self.size = size
        groups = defaultdict(list)
        for i, row in enumerate(raw_codes.tolist()):
            groups[tuple(row)].append(i)
        max_bucket = max(map(len, groups.values()))
        digits = max(1, math.ceil(math.log(max_bucket, size)))
        while size**digits < max_bucket:
            digits += 1
        suffix = np.zeros((len(raw_codes), digits), dtype=int)
        for members in groups.values():
            for rank, item in enumerate(members):
                for digit in range(digits):
                    suffix[item, digits - digit - 1] = (rank // size**digit) % size
        full = np.concatenate([raw_codes, suffix], axis=1)
        self.codes = full + 3 + np.arange(full.shape[1])[None] * size
        self.vocab = 3 + full.shape[1] * size
        self.depth = full.shape[1]
        self.children = defaultdict(set)
        self.leaves = {}
        for item, row in enumerate(self.codes.tolist()):
            for d, token in enumerate(row):
                self.children[tuple(row[:d])].add(token)
            self.leaves[tuple(row)] = item
        self.diagnostics = {
            "semantic_collision_rate": 1 - len(groups) / len(raw_codes),
            "max_collision_bucket": max_bucket,
            "suffix_digits": digits,
            "final_collision_rate": 0.0,
            "utilization": [
                len(set(raw_codes[:, l])) / size for l in range(raw_codes.shape[1])
            ],
            "perplexity": [],
        }
        for col in raw_codes.T:
            p = np.array(list(Counter(col).values())) / len(col)
            self.diagnostics["perplexity"].append(float(np.exp(-(p * np.log(p)).sum())))

    def history_tokens(self, history, max_history):
        result = [self.BOS]
        for item in history[-max_history:]:
            result.extend(self.codes[item].tolist())
        return result + [self.SEP]

    @torch.no_grad()
    def beam_search(self, model, history, k=12, beam=32, max_history=30):
        if beam < k:
            raise ValueError("beam must be >= k")
        model.eval()
        device = next(model.parameters()).device
        context = self.history_tokens(history, max_history)
        candidates = [((), 0.0)]
        for _ in range(self.depth):
            batch = torch.tensor(
                [context + list(prefix) for prefix, _ in candidates], device=device
            )
            # Normalize over full vocabulary, THEN mask invalid trie edges: preserve model probabilities.
            log_probs = model(batch)[:, -1].log_softmax(-1)
            expanded = []
            for row, (prefix, score) in enumerate(candidates):
                for token in sorted(self.children[prefix]):
                    expanded.append(
                        (prefix + (token,), score + float(log_probs[row, token]))
                    )
            candidates = sorted(expanded, key=lambda pair: (-pair[1], pair[0]))[:beam]
        return [self.leaves[prefix] for prefix, _ in candidates[:k]]
