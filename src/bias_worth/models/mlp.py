"""Low-bias model B: a plain MLP on the flattened input.

Follows Bachmann et al. (2023): pre-norm residual blocks with an inverted
bottleneck, and no knowledge of the input's spatial structure.
"""

import math

import torch


class _Block(torch.nn.Module):
    def __init__(self, width: int, expansion: int) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(width)
        self.up = torch.nn.Linear(width, width * expansion)
        self.down = torch.nn.Linear(width * expansion, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = x + self.down(
            torch.nn.functional.gelu(self.up(self.norm(x)))
        )
        return out


class MLP(torch.nn.Module):
    """A residual bottleneck MLP that flattens its input first."""

    def __init__(
        self,
        input_shape: tuple[int, ...],
        num_classes: int,
        width: int = 512,
        depth: int = 4,
        expansion: int = 4,
    ) -> None:
        """Builds the MLP.

        Args:
            input_shape: Shape of one example, without the batch axis.
            num_classes: Output classes.
            width: Residual stream width.
            depth: Number of residual blocks.
            expansion: Hidden width multiplier inside each block.
        """
        super().__init__()
        self.embed = torch.nn.Linear(math.prod(input_shape), width)
        self.blocks = torch.nn.Sequential(
            *[_Block(width, expansion) for _ in range(depth)]
        )
        self.norm = torch.nn.LayerNorm(width)
        self.head = torch.nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classifies a batch.

        Args:
            x: Inputs of shape (batch, *input_shape).

        Returns:
            Logits of shape (batch, num_classes).
        """
        h = self.blocks(self.embed(x.flatten(1)))
        out: torch.Tensor = self.head(self.norm(h))
        return out
