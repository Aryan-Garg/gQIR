import torch
import torch.nn as nn


# TODO
class N2N(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @torch.no_grad()
    def load_pretrained(self, sd: dict) -> None:
        self.model.load_state_dict(sd, strict=True)