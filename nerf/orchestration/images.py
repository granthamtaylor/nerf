import flytekit as fk

image = fk.ImageSpec(
    registry="ghcr.io/granthamtaylor",
    name='byoc-sandbox',
    requirements="requirements.txt",
    apt_packages=["build-essential"],
)

wandb_secret = fk.Secret(key="WANDB_API_KEY")