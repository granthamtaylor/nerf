import typing
import itertools

from collections import OrderedDict
from nerf.core.structs import Hyperparameters
import dataclasses


def permutate_params(
    base: Hyperparameters,
    d_model: typing.Optional[typing.Iterable[int]] = None,
    n_bands: typing.Optional[typing.Iterable[int]] = None,
    n_layers: typing.Optional[typing.Iterable[int]] = None,
    offset: typing.Optional[typing.Iterable[int]] = None,
    learning_rate: typing.Optional[typing.Iterable[float]] = None,
) -> typing.List[Hyperparameters]:
    """
    Helper function to generate a list of Hyperparameters for testing & optimizing.
    Runs at compile time, this is not a task. Output is hard-coded as static input to task or workflow.
    """
    params = {
        "d_model": d_model,
        "n_bands": n_bands,
        "n_layers": n_layers,
        "offset": offset,
        "learning_rate": learning_rate,
    }
    params = OrderedDict({k: v for k, v in params.items() if v is not None})

    all_params = list(params.values())

    altered_keys = list(params.keys())
    deltas = itertools.product(*all_params)
    x = [dd for dd in deltas]
    assert all(len(altered_keys) == len(delta) for delta in deltas)
    replacements = [dict(zip(altered_keys, delta)) for delta in x]

    # ignore type, pydantic dataclass is compatible with dataclasses but not recognized as such.
    return [dataclasses.replace(base, **d) for d in replacements]  # type: ignore
