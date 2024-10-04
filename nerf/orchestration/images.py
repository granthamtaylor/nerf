import flytekit as fk

image = fk.ImageSpec(
    builder="union",
    requirements="requirements.txt",
    apt_packages=["build-essential"],
)
