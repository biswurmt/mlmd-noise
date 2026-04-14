.PHONY: setup install-medmnist smoke-medmnist smoke-tabular

VENV=.venv
PY=${VENV}/bin/python

setup:
	./scripts/setup_env.sh

install-medmnist:
	./scripts/setup_env.sh --with-medmnist

smoke-medmnist:
	@if [ -x "${PY}" ]; then \
	  ${PY} scripts/run_medmnist_matrix.py --datasets bloodmnist --noise-rates 0.0 --seeds 42 --epochs 2 --output-dir results/smoke/medmnist_blood_e2; \
	else \
	  echo "Virtualenv not found. Run 'make setup' first."; exit 1; \
	fi

smoke-tabular:
	@if [ -x "${PY}" ]; then \
	  ${PY} scripts/run_tabular_matrix.py --datasets adult --noise-rates 0.2 --seeds 42 --epochs 2 --output-dir results/smoke/tabular_adult_e2; \
	else \
	  echo "Virtualenv not found. Run 'make setup' first."; exit 1; \
	fi
