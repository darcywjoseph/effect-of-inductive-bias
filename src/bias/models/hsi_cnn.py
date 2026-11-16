"""Domain-specific hyperspectral model: a spectral-spatial CNN.

3-D convolutions run over bands and space together, then the band axis is
folded into channels for 2-D convolutions, as in HybridSN (Roy et al.,
2020). Spatial padding keeps the small 5x5 patch from vanishing.
"""

import torch


class SpectralSpatialCNN(torch.nn.Module):
    """3-D then 2-D convolutions over a (bands, side, side) patch."""

    def __init__(
        self, bands: int, side: int, num_classes: int, width: int = 128
    ) -> None:
        """Builds the network.

        Args:
            bands: Spectral bands per pixel.
            side: Patch height and width.
            num_classes: Output classes.
            width: Channels of the 2-D convolution; set per scene so the
              parameter count stays near the other models'.
        """
        super().__init__()

        def conv3d(cin: int, cout: int, depth: int) -> torch.nn.Sequential:
            return torch.nn.Sequential(
                torch.nn.Conv3d(cin, cout, (depth, 3, 3), padding=(0, 1, 1)),
                torch.nn.BatchNorm3d(cout),
                torch.nn.ReLU(inplace=True),
            )

        self.spectral = torch.nn.Sequential(
            conv3d(1, 8, 7), conv3d(8, 16, 5), conv3d(16, 32, 3)
        )
        folded = 32 * (bands - 12)
        self.spatial = torch.nn.Sequential(
            torch.nn.Conv2d(folded, width, 3, padding=1),
            torch.nn.BatchNorm2d(width),
            torch.nn.ReLU(inplace=True),
        )
        self.num_features = 256
        self.neck = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(width * side * side, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.4),
        )
        self.head: torch.nn.Module = torch.nn.Linear(256, num_classes)

    def reset_classifier(self, num_classes: int) -> None:
        """Replaces the head, as timm models do; 0 removes it.

        Args:
            num_classes: New output classes, or 0 to output features.
        """
        self.head = (
            torch.nn.Linear(self.num_features, num_classes)
            if num_classes
            else torch.nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classifies a batch of patches.

        Args:
            x: Patches of shape (batch, bands, side, side).

        Returns:
            Logits of shape (batch, num_classes).
        """
        h = self.spectral(x.unsqueeze(1))
        h = h.flatten(1, 2)
        out: torch.Tensor = self.head(self.neck(self.spatial(h)))
        return out
