import typing
from functools import partial

import flytekit as fk
from flytekit.types.file import FlyteFile

from nerf.tasks.fit import show_file, hello_world


@fk.workflow
def train(
    image: FlyteFile,
):
    show_file(img=image)


@fk.workflow
def call_hello():
    hello_world()


if __name__ == "__main__":
    train()
