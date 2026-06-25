SCRATCH_DIR := /scratch/$(USER)
export TMPDIR ?= $(SCRATCH_DIR)/tmp
export PIP_CACHE_DIR ?= $(SCRATCH_DIR)/pip_cache
export UV_CACHE_DIR ?= $(SCRATCH_DIR)/uv_cache
export XDG_CACHE_HOME ?= $(SCRATCH_DIR)/.cache
export TORCH_EXTENSIONS_DIR ?= $(SCRATCH_DIR)/torch_extensions
export CUDA_CACHE_PATH ?= $(SCRATCH_DIR)/.nv/ComputeCache

i install:
	@python -c "import sys; sys.exit(0 if (sys.prefix != sys.base_prefix) else 1)" || \
		(echo "ERROR: active python ($$(which python)) is not inside a venv -- run 'conda deactivate' (if conda is active) and activate the venv before 'make i'" && exit 1)
	mkdir -p "$(TMPDIR)" "$(PIP_CACHE_DIR)" "$(UV_CACHE_DIR)" "$(XDG_CACHE_HOME)" "$(TORCH_EXTENSIONS_DIR)" "$(CUDA_CACHE_PATH)"
	cd csrc && make
	@if python -c "import pip" >/dev/null 2>&1; then \
		python -m pip install --no-build-isolation -e . ; \
	else \
		uv pip install --no-build-isolation -e . ; \
	fi
