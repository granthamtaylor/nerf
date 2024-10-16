from typing import Annotated

from flytekit.types.file import FlyteFile

from pydantic.dataclasses import dataclass
from pydantic import Field
from tensordict import tensorclass
import torch

MAXBYTEVALUE = 255


@tensorclass
class CoordinateTensor:
    """A tensor of XY coordinates"""

    x: torch.Tensor
    y: torch.Tensor

    @property
    def xy(self) -> torch.Tensor:
        return torch.cat((self.x, self.y), dim=1)


@tensorclass
class PixelTensor:
    """A tensor of RGB pixel values"""

    red: torch.Tensor
    green: torch.Tensor
    blue: torch.Tensor

    @property
    def rgb(self) -> torch.Tensor:
        return torch.cat((self.red, self.green, self.blue), dim=1)

    @property
    def normalized(self) -> torch.Tensor:
        return self.rgb.mul(1 / MAXBYTEVALUE)

    @classmethod
    def convert(self, predictions: torch.Tensor) -> torch.Tensor:
        return predictions.mul(MAXBYTEVALUE).type(torch.int32)

@tensorclass
class InputTensor:
    """A tensor of input data"""

    coordinates: CoordinateTensor
    color: PixelTensor

@dataclass
class SearchSpace:
    """A search space for hyperparameters"""

    batch_size: list[int]
    d_model: list[int]
    n_bands: list[int]
    n_layers: list[int]
    offset: list[int]
    max_epochs: list[int]
    learning_rate: list[float]
    patience: list[int]


@dataclass
class Hyperparameters:
    """Hyperparameters for the model"""

    batch_size: Annotated[int, Field(gt=0, le=2048)] = 256
    d_model: Annotated[int, Field(gt=1, le=256)] = 64
    n_bands: Annotated[int, Field(gt=1, le=16)] = 8
    n_layers: Annotated[int, Field(ge=1, le=64)] = 4
    offset: Annotated[int, Field(ge=1, le=512)] = 2
    max_epochs: Annotated[int, Field(ge=1, le=1028)] = 16
    learning_rate: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.001
    patience: Annotated[int, Field(ge=1, le=256)] = 16


@dataclass
class Result:
    """The result of a trained model"""

    animation: FlyteFile
    model: FlyteFile
    params: Hyperparameters

@dataclass
class Metric:
    """Metrics for the model's performance"""

    loss: float
    compression: float