# run workflow locally
dev:
  @poetry run union run nerf/workflows/train.py train --image='papas10.jpg'

# run workflow on remote
run:
  @poetry run union run --remote --copy-all nerf/workflows/train.py train --image='papas10.jpg'

# register workflow to remote
register: freeze
  @poetry run union register nerf

freeze:
  @poetry run uv pip compile pyproject.toml > requirements.txt --python-platform linux
