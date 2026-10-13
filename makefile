SHELL := /bin/bash

# Build the varlen per-head KV-cache kernel (csrc/ -> tiny_api_cuda) and install
# this repository in editable mode into the active environment. Machine-specific
# paths (scratch caches, compiler modules) can be set in .env; see .env.example.
LOAD_ENV = set -a; if [ -f .env ]; then . ./.env; fi; set +a;

install i:
	@python -c "import sys; sys.exit(sys.prefix == sys.base_prefix)" || \
		{ echo "error: activate a virtual environment first (see README: Installation)"; exit 1; }
	@$(LOAD_ENV) cd csrc && python build.py install
	@$(LOAD_ENV) if python -c "import pip" >/dev/null 2>&1; then \
		python -m pip install --no-build-isolation -e . ; \
	else \
		uv pip install --no-build-isolation -e . ; \
	fi

test:
	python -m pytest -q tests

lint:
	ruff check --select F lorekv scripts experiments models_ext tests

.PHONY: install i test lint
