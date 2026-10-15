UV ?= $(HOME)/.local/bin/uv

.PHONY: check test lint fmt hft-test

check: lint test hft-test

lint:
	$(UV) run ruff check src tests

fmt:
	$(UV) run ruff check --fix src tests

test:
	$(UV) run pytest -W ignore::DeprecationWarning

hft-test:
	cd hft && $(UV) run python -m pytest
