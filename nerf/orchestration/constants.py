import flytekit

image = flytekit.ImageSpec(
    registry="ghcr.io/granthamtaylor",
    name='byoc-sandbox',
    requirements="requirements.txt",
    apt_packages=["build-essential"],
)

wandb_secret = flytekit.Secret(key="WANDB_API_KEY")