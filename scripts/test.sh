set -e
set -x

bash ./scripts/lint.sh
# Use xdist-pytest --forked to ensure modified sys.path to import relative modules in examples keeps working
pytest -m unittest