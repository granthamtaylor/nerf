# run workflow locally
dev:
  @uv run union run nerf/workflows/train.py train --image='pie.jpg'

# run workflow on remote
run: freeze
  @uv run union run --remote nerf/workflows/train.py train

# register workflow to remote
register: freeze
  @uv run union register nerf

freeze:
  @uv pip compile pyproject.toml > requirements.txt --python-platform linux

reset:
  @uv run pyflyte local-cache clear