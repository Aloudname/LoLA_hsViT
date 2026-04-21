from __future__ import annotations

import torch
import torch.nn as nn


class BaseBackbone(nn.Module):
    """Unified token backbone interface.

    Input:
        tokens: [B, N, D]
    Output:
        tokens: [B, N, D]

    Notes:
        - D is expected to match the backbone hidden dimension.
        - If the model's token dimension differs from backbone hidden dimension,
          explicit projection layers should be applied inside the backbone wrapper.
    """

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface only
        raise NotImplementedError
