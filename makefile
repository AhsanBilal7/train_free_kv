SHELL := /bin/bash
ENV_FILE ?= .env

# Machine-specific paths live in $(ENV_FILE), the same file scripts/*.sh source
# (see .env). It is bash -- conditionals, ${VAR:-default} -- so make cannot
# `include` it; every recipe sources it instead, under `set -a` so the values
# are exported to python/pip/nvcc. The `: "$${VAR:=...}"` lines below are pure
# fallbacks for a missing .env and mirror its defaults; anything already set in
# the environment or in .env wins.
LOAD_ENV = set -a; \
	if [ -f "$(ENV_FILE)" ]; then . "$(ENV_FILE)"; fi; \
	: "$${SCRATCH:=$$(test -d /scratch/user/$$USER && echo /scratch/user/$$USER || echo /scratch/$$USER)}"; \
	: "$${TMPDIR:=$$SCRATCH/tmp}"; \
	: "$${PIP_CACHE_DIR:=$$SCRATCH/pip_cache}"; \
	: "$${UV_CACHE_DIR:=$$SCRATCH/uv_cache}"; \
	: "$${XDG_CACHE_HOME:=$$SCRATCH/.cache}"; \
	: "$${TORCH_EXTENSIONS_DIR:=$$SCRATCH/torch_extensions}"; \
	: "$${CUDA_CACHE_PATH:=$$SCRATCH/.nv/ComputeCache}"; \
	set +a;

i install:
	@$(LOAD_ENV) python -c "import sys; sys.exit(0 if (sys.prefix != sys.base_prefix) else 1)" || \
		(echo "ERROR: active python ($$(which python)) is not inside a venv -- run 'conda deactivate' (if conda is active) and activate the venv before 'make i'" && exit 1)
	@$(LOAD_ENV) mkdir -p "$$TMPDIR" "$$PIP_CACHE_DIR" "$$UV_CACHE_DIR" "$$XDG_CACHE_HOME" "$$TORCH_EXTENSIONS_DIR" "$$CUDA_CACHE_PATH"
	@$(LOAD_ENV) cd csrc && make
	@$(LOAD_ENV) if python -c "import pip" >/dev/null 2>&1; then \
		python -m pip install --no-build-isolation -e . ; \
	else \
		uv pip install --no-build-isolation -e . ; \
	fi

# Print the paths a recipe would use -- run this when a build writes somewhere
# unexpected: make env
env:
	@$(LOAD_ENV) for v in SCRATCH TMPDIR PIP_CACHE_DIR UV_CACHE_DIR XDG_CACHE_HOME \
		TORCH_EXTENSIONS_DIR CUDA_CACHE_PATH HF_HOME VENV_PATH UV_PROJECT_ENVIRONMENT; do \
		printf '%-24s %s\n' "$$v" "$${!v:-<unset>}"; \
	done

.PHONY: i install env
