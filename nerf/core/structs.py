from typing import Annotated

from pydantic.dataclasses import dataclass
from pydantic import Field
from tensordict import tensorclass
import torch

MAXBYTEVALUE = 255


@tensorclass
class CoordinateTensor:
    x: torch.Tensor
    y: torch.Tensor

    @property
    def xy(self) -> torch.Tensor:
        return torch.cat((self.x, self.y), dim=1)


@tensorclass
class PixelTensor:
    red: torch.Tensor
    green: torch.Tensor
    blue: torch.Tensor
    aaa: torch.Tensor

    @property
    def rgb(self) -> torch.Tensor:
        return torch.cat((self.red, self.green, self.blue, self.aaa), dim=1)

    @property
    def normalized(self) -> torch.Tensor:
        return self.rgb.mul(1 / MAXBYTEVALUE)

    @classmethod
    def convert(cls, predictions: torch.Tensor) -> torch.Tensor:
        return predictions.mul(MAXBYTEVALUE).type(torch.int32)


@tensorclass
class InputTensor:
    coordinates: CoordinateTensor
    color: PixelTensor


@dataclass
class Hyperparameters:
    batch_size: Annotated[int, Field(gt=0, le=2048)] = 256
    n_workers: Annotated[int, Field(ge=0, le=256)] = 4
    d_model: Annotated[int, Field(gt=1, le=256)] = 64
    n_bands: Annotated[int, Field(gt=1, le=16)] = 8
    n_layers: Annotated[int, Field(gt=1, le=64)] = 4
    offset: Annotated[int, Field(gt=0, le=512)] = 2
    max_epochs: Annotated[int, Field(gt=0, le=1028)] = 4
    learning_rate: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.001
    patience: Annotated[int, Field(ge=1, le=256)] = 16
