# run workflow locally
dev:
  @uv run union run nerf/workflows/train.py train --image='images/pie.jpg'

# run workflow on remote
run: freeze
  @uv run union run --remote --copy auto nerf/workflows/train.py train

# register workflow to remote
register: freeze
  @uv run union register nerf

freeze:
  @uv pip compile pyproject.toml > requirements.txt --python-platform linux
