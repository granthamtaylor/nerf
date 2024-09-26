from nerf.core.structs import Hyperparameters
from nerf.core.utils import permutate_params


def test_permute():
    new_params = permutate_params(
        Hyperparameters(),
        d_model=[64, 128, 256],
    )
    assert len(new_params) == 3
    assert new_params[0].d_model == 64
    assert new_params[1].d_model == 128
    assert new_params[2].d_model == 256


def test_permute_product():
    new_params = permutate_params(
        Hyperparameters(),
        d_model=[64, 128],
        n_bands=[7, 8, 9],
        n_layers=[4, 8],
        offset=[1, 2],
        learning_rate=[0.001, 0.005, 0.01],
    )

    assert len(new_params) == 72

