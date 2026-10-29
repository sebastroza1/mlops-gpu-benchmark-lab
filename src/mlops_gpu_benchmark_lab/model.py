from __future__ import annotations

from collections.abc import Sequence

from torch import nn

ARCHITECTURES: dict[str, tuple[int, ...]] = {
    "small": (64, 32),
    "medium": (128, 64, 32),
}


class CovertypeMLP(nn.Module):
    def __init__(
        self,
        input_features: int = 54,
        hidden_sizes: Sequence[int] = ARCHITECTURES["small"],
        classes: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_features
        for width in hidden_sizes:
            layers.extend((nn.Linear(current, width), nn.ReLU(), nn.Dropout(dropout)))
            current = width
        layers.append(nn.Linear(current, classes))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return self.network(features)


def create_model(name: str, input_features: int = 54) -> CovertypeMLP:
    try:
        hidden_sizes = ARCHITECTURES[name]
    except KeyError as exc:
        choices = sorted(ARCHITECTURES)
        raise ValueError(f"Unknown architecture {name!r}; choose from {choices}") from exc
    return CovertypeMLP(input_features=input_features, hidden_sizes=hidden_sizes)
