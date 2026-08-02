SHELL := /bin/bash

# Folders that hold course code, in reading order. Update this list if a folder is
# added/renamed — every target below (lint-<dir>, fix-<dir>, ...) derives from it.
DIRS := 01_langchain_fundamentals 02_langgraph_control_flow 03_rag_and_memory \
        04_multi_agent_systems 05_production_patterns projects

.DEFAULT_GOAL := help

.PHONY: help install sync \
        lint lint-fix format format-check check compile clean \
        $(addprefix lint-,$(DIRS)) $(addprefix fix-,$(DIRS)) $(addprefix format-,$(DIRS))

help: ## Show this help
	@echo "Usage: make <target> [DIR=<folder>]"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Folders: $(DIRS)"
	@echo "Examples:"
	@echo "  make lint                          # ruff check, whole project"
	@echo "  make lint DIR=03_rag_and_memory     # ruff check, one folder"
	@echo "  make lint-03_rag_and_memory         # same, as a named target"
	@echo "  make fix DIR=check_api_connection.py  # ruff --fix, one file"

# --- environment ---

install: ## Install/sync dependencies from pyproject.toml / uv.lock
	uv sync

sync: install

# --- whole-project quality gates ---
# DIR=<path> narrows any of these to one folder or file, e.g. `make lint DIR=projects`.

lint: ## Ruff lint (add DIR=<path> to scope it)
	uv run ruff check $(or $(DIR),.)

lint-fix: ## Ruff lint with autofix (add DIR=<path> to scope it)
	uv run ruff check $(or $(DIR),.) --fix

fix: lint-fix

format: ## Ruff format, i.e. auto-format code (add DIR=<path> to scope it)
	uv run ruff format $(or $(DIR),.)

format-check: ## Check formatting without writing changes (add DIR=<path> to scope it)
	uv run ruff format $(or $(DIR),.) --check

compile: ## Byte-compile every .py file (fast syntax-error check)
	@find . -name '*.py' -not -path './.venv/*' -print0 | xargs -0 -n1 python3 -m py_compile

check: lint format-check compile ## Run every check (lint + format-check + compile) — use before committing

clean: ## Remove caches and run artifacts (__pycache__, .ruff_cache, demo DBs/PNGs)
	find . -name '__pycache__' -not -path './.venv/*' -exec rm -rf {} +
	rm -rf .ruff_cache
	rm -rf chroma_db research_db
	rm -f chat_history.db

# --- per-folder convenience targets ---
# Same as `make lint DIR=<folder>` but tab-completable, e.g. `make lint-04_multi_agent_systems`.

$(addprefix lint-,$(DIRS)):
	uv run ruff check $(subst lint-,,$@)

$(addprefix fix-,$(DIRS)):
	uv run ruff check $(subst fix-,,$@) --fix

$(addprefix format-,$(DIRS)):
	uv run ruff format $(subst format-,,$@)
