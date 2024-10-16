import flytekit

from nerf.orchestration.environments import TaskEnvironment, EnvironmentContext

env = TaskEnvironment(
    name='basic',
    container_image=flytekit.ImageSpec(
        registry="ghcr.io/granthamtaylor",
        name='byoc-sandbox',
        requirements="requirements.txt",
        apt_packages=["build-essential"],
    ),
    secret_requests=[flytekit.Secret(key="WANDB_API_KEY")],
    cache=True,
    cache_version="#cache-v1",
)

context = EnvironmentContext(
    env,
    env.extend(name='gpu', requests=flytekit.Resources(gpu="1", cpu="16", mem="64Gi")),
)