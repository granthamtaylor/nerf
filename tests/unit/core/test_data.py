import random
import os
import itertools
import polars as pl

from PIL import Image
import numpy as np
import torch

from tensordict import tensorclass


@tensorclass
class PNGTensor:
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


def test_xjljkl():

    with Image.open("/Users/ytong/Downloads/ufo.png") as file:
        data = np.array(file)

    x, y, C = shape = data.shape
    rgba = torch.tensor(data[0, 0], dtype=torch.int32).split(1, dim=0)
    aa = data[:, :, 3]
    print(aa)

    # print(f"shape: {shape}")
    print(rgba)
