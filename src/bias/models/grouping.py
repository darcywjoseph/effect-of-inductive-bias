"""A ViT that groups pixels into its tokens instead of cutting fixed patches.

Each of the 1,024 pixels is embedded on its own (its RGB values plus a
position vector), so the embedding knows nothing about which pixels are
neighbours. The pixels are then pooled into GROUPS tokens, which go
through the plain ViT-Tiny's blocks and head.

Kinds:

- "learned": soft grouping, slot-attention style. Learned slots score
  every pixel, each pixel's scores are softmaxed over the groups, each
  group token is the weighted mean of its pixels, and the slots are
  refined from those means for ITERS rounds. Positions are learned from a
  random start, so the model is blind to any fixed reordering of the
  pixels and has to find out from data which pixels belong together.
- "grid": the same pixel embedding, pooled over fixed 4x4 blocks. Controls
  for the machinery: it has the plain ViT's grouping.
- "fixedpos": "learned", plus fixed 2-D sinusoidal positions added to the
  pixel embedding, so the grouping can use real image coordinates if it
  helps.
"""

import math

import torch

from bias.models import vit

SIDE = vit.IMG_SIZE
PIXELS = SIDE * SIDE
GROUPS = 64
ITERS = 3
KINDS = ("learned", "grid", "fixedpos")


def grid_assignment(side: int = SIDE, block: int = vit.PATCH_SIZE) -> torch.Tensor:
    """Returns the one-hot assignment of every pixel to its block.

    Args:
        side: Pixels per side of the image.
        block: Pixels per side of a block; must divide side.

    Returns:
        A (side * side, (side // block) ** 2) tensor with one 1 per row.
    """
    rows, cols = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    per_side = side // block
    group = (rows // block) * per_side + cols // block
    return torch.nn.functional.one_hot(group.reshape(-1), per_side**2).float()


class GroupViT(torch.nn.Module):
    """Pixel embedding, then grouping into tokens, then the ViT-Tiny core."""

    def __init__(
        self, num_classes: int, kind: str = "learned", groups: int = GROUPS
    ) -> None:
        """Builds the model.

        Args:
            num_classes: Output classes.
            kind: One of KINDS.
            groups: Number of group tokens; must be 64 for "grid".

        Raises:
            ValueError: If the kind is unknown or "grid" has another count.
        """
        super().__init__()
        if kind not in KINDS or (kind == "grid" and groups != GROUPS):
            raise ValueError(f"unsupported grouping: {kind=}, {groups=}")
        self.kind = kind
        core = vit.vit_tiny(num_classes)
        width: int = core.embed_dim  # type: ignore[assignment]
        self.width = width
        self.cls_token: torch.nn.Parameter = core.get_parameter("cls_token")
        self.blocks = core.get_submodule("blocks")
        self.norm = core.get_submodule("norm")
        self.head = core.get_submodule("head")

        self.pixel_in = torch.nn.Linear(3, width)
        self.pixel_pos = torch.nn.Parameter(torch.zeros(1, PIXELS, width))
        torch.nn.init.trunc_normal_(self.pixel_pos, std=0.02)
        self.pixel_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, 2 * width),
            torch.nn.GELU(),
            torch.nn.Linear(2 * width, width),
        )
        self.fixed_pos: torch.Tensor
        fixed = vit.sincos_2d(SIDE, width) if kind == "fixedpos" else torch.zeros(1)
        self.register_buffer("fixed_pos", fixed, persistent=False)
        self.grid: torch.Tensor
        grid = grid_assignment() if kind == "grid" else torch.zeros(1)
        self.register_buffer("grid", grid, persistent=False)
        self.slots = torch.nn.Parameter(torch.randn(groups, width) * 0.02)
        self.norm_pixel = torch.nn.LayerNorm(width)
        self.norm_slot = torch.nn.LayerNorm(width)
        self.to_k = torch.nn.Linear(width, width, bias=False)
        self.to_q = torch.nn.Linear(width, width, bias=False)

    def pixel_features(self, x: torch.Tensor) -> torch.Tensor:
        """Embeds every pixel on its own.

        Args:
            x: Images of shape (batch, 3, SIDE, SIDE).

        Returns:
            Features of shape (batch, PIXELS, width).
        """
        f = self.pixel_in(x.flatten(2).transpose(1, 2)) + self.pixel_pos
        if self.kind == "fixedpos":
            f = f + self.fixed_pos
        out: torch.Tensor = f + self.pixel_mlp(f)
        return out

    def assignments(self, f: torch.Tensor) -> torch.Tensor:
        """Returns each pixel's soft membership of each group.

        Args:
            f: Pixel features from pixel_features().

        Returns:
            A (batch, PIXELS, groups) tensor whose rows sum to one.
        """
        if self.kind == "grid":
            return self.grid.to(f.dtype).expand(len(f), -1, -1)
        keys = self.to_k(self.norm_pixel(f))
        slots = self.slots.expand(len(f), -1, -1)
        for _ in range(ITERS):
            queries = self.to_q(self.norm_slot(slots))
            scores = keys @ queries.transpose(1, 2) / math.sqrt(self.width)
            weights: torch.Tensor = scores.softmax(dim=-1)
            slots = _pool(weights, f)
        return weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classifies a batch of images.

        Args:
            x: Images of shape (batch, 3, SIDE, SIDE).

        Returns:
            Logits of shape (batch, num_classes).
        """
        f = self.pixel_features(x)
        tokens = _pool(self.assignments(f), f)
        cls = self.cls_token.expand(len(x), -1, -1)
        h = self.norm(self.blocks(torch.cat([cls, tokens], dim=1)))
        out: torch.Tensor = self.head(h[:, 0])
        return out


def _pool(weights: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Returns each group's weighted mean of the pixel features."""
    share = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
    return share.transpose(1, 2) @ f
