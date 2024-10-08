import math
import os
from functools import partialmethod
from collections import OrderedDict

import torch
import lightning.pytorch as lit
from torch.utils.data import DataLoader

from nerf.core.structs import InputTensor, Hyperparameters, PixelTensor
from nerf.core.data import BitMapIterator, collate


class FourierEncoder(torch.nn.Module):
    def __init__(self, params: Hyperparameters) -> None:
        super().__init__()

        weights = torch.logspace(
            start=-params.n_bands, end=params.offset, steps=params.n_bands + params.offset + 1, base=2
        )

        self.linear = torch.nn.Linear(2 * len(weights), params.d_model)
        self.register_buffer("weights", weights.mul(math.pi).unsqueeze(dim=0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weighted = inputs.mul(self.weights)

        fourier = torch.sin(weighted), torch.cos(weighted)

        projections = self.linear(torch.cat(fourier, dim=1))

        return projections


class MLP(torch.nn.Module):
    def __init__(self, params: Hyperparameters):
        super().__init__()

        C = 3

        layers = []

        combiner = torch.nn.Sequential(
            OrderedDict(
                [
                    ("dense", torch.nn.Linear(params.d_model * 2, params.d_model)),
                    ("act", torch.nn.ReLU()),
                ]
            )
        )

        for _ in range(params.n_layers):
            layer = torch.nn.Sequential(
                OrderedDict(
                    [
                        ("dense", torch.nn.Linear(params.d_model, params.d_model)),
                        ("act", torch.nn.ReLU()),
                    ]
                )
            )

            layers.append(layer)

        body = torch.nn.Sequential(*layers)

        head = torch.nn.Sequential(
            OrderedDict(
                [
                    ("dense", torch.nn.Linear(params.d_model, C)),
                    ("act", torch.nn.Sigmoid()),
                ]
            )
        )

        self.model = torch.nn.Sequential(
            OrderedDict(
                [
                    ("combiner", combiner),
                    ("body", body),
                    ("head", head),
                ]
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


def dataloader(self: "NeRFModule", strata: str) -> DataLoader:
    assert strata in ["train", "validate", "test", "predict"]

    pipe = self.pipes[strata]
    self.dataloaders[strata] = loader = DataLoader(
        pipe,
        batch_size=self.params.batch_size,
        num_workers=os.cpu_count(),
        collate_fn=collate,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False,
    )

    return loader


def step(self: "NeRFModule", inputs: InputTensor, strata: str) -> dict[str, torch.Tensor]:
    assert strata in ["train", "validate", "test", "predict"]

    predictions = self.forward(inputs.coordinates)

    loss = torch.nn.functional.mse_loss(predictions, inputs.color.normalized)

    self.log(f"{strata}/loss", loss)

    return {"loss": loss, "predictions": PixelTensor.convert(predictions)}


# lightning stage to strata mapping
MAPPING: dict[str, list[str]] = dict(
    fit=["train", "validate"],
    validate=["validate"],
    test=["test"],
    predict=["predict"],
)


class NeRFModule(lit.LightningModule):
    def __init__(
        self,
        params: Hyperparameters,
        image: os.PathLike,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(vars(params))
        self.params = params
        self.image = image

        self.horizontal_encoder = FourierEncoder(params)
        self.vertical_encoder = FourierEncoder(params)
        self.mlp = MLP(params)

        self.dataloaders: dict[str, DataLoader] = {}
        self.pipes: dict[str, BitMapIterator] = {}

    def __len__(self) -> int:
        """
        Returns the number bytesize of model.
        """

        out = 0
        for param in self.parameters():
            out += param.nelement() * param.element_size()

        for buffer in self.buffers():
            out += buffer.nelement() * buffer.element_size()

        return out

    def forward(self, inputs: InputTensor) -> torch.Tensor:
        x: torch.Tensor = self.horizontal_encoder(inputs.x)
        y: torch.Tensor = self.vertical_encoder(inputs.y)

        combined: torch.Tensor = torch.cat((x, y), dim=1)

        return self.mlp(combined)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.params.learning_rate)

    training_step = partialmethod(step, strata="train")  # type: ignore
    validation_step = partialmethod(step, strata="validate")  # type: ignore
    test_step = partialmethod(step, strata="test")  # type: ignore
    predict_step = partialmethod(step, strata="predict")  # type: ignore

    train_dataloader = partialmethod(dataloader, strata="train")  # type: ignore
    val_dataloader = partialmethod(dataloader, strata="validate")  # type: ignore
    test_dataloader = partialmethod(dataloader, strata="test")  # type: ignore
    predict_dataloader = partialmethod(dataloader, strata="predict")  # type: ignore

    def setup(self, stage: str):
        assert stage in ["fit", "validate", "test", "predict"]

        for strata in MAPPING[stage]:
            self.pipes[strata] = BitMapIterator(image=self.image)

    def teardown(self, stage: str):
        assert stage in ["fit", "validate", "test", "predict"]

        for strata in MAPPING[stage]:
            del self.pipes[strata]
            del self.dataloaders[strata]
