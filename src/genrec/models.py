import torch
from torch import nn


class CausalTransformer(nn.Module):
    """Decoder-only next-token model, also used as an item-ID sequence baseline."""

    def __init__(self, vocab, max_length, dim=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.token = nn.Embedding(vocab, dim, padding_idx=0)
        self.position = nn.Embedding(max_length, dim)
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout, batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.token.weight

    def forward(self, ids):
        if ids.size(1) > self.position.num_embeddings:
            raise ValueError("sequence exceeds configured maximum length")
        position = torch.arange(ids.size(1), device=ids.device)
        x = self.token(ids) + self.position(position)
        mask = torch.triu(
            torch.ones(ids.size(1), ids.size(1), dtype=torch.bool, device=ids.device),
            diagonal=1,
        )
        return self.head(
            self.norm(self.blocks(x, mask=mask, src_key_padding_mask=ids.eq(0)))
        )
