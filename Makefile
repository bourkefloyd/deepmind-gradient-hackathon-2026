# Word Hunt arena - nano training + local Gemma understudy. Targets check-mps and mlxvlm-* are ported from
# actionfleet's Makefile (same model tag and flags).

.PHONY: help setup words check-mps data-smoke train-smoke rollout-smoke data train mlxvlm-up mlxvlm-down mlxvlm-stop mlxvlm-status

SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= .venv/bin/python

# Local vision Gemma 4 12B served by mlx-vlm (Apple Silicon). Point MLXVLM_MODEL at a HF repo or local path.
MLXVLM_MODEL ?= mlx-community/gemma-4-12B-it-4bit
MLXVLM_PORT  ?= 8080
MLXVLM_HOST  ?= http://localhost:8080

# nano knobs
DEPTH   ?= 6
STEPS   ?= 20000
BOARDS  ?= 200000
DATA    ?= data/wh_$(BOARDS).npz
RUN     ?= runs/d$(DEPTH)_s0

CYAN   := \033[1;36m
YELLOW := \033[1;33m
GREEN  := \033[1;32m
RED    := \033[1;31m
MAGENTA:= \033[1;35m
RESET  := \033[0m

help: ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-16s$(RESET) %s\n", $$1, $$2}'

setup: ## Create .venv with torch + numpy (uv if present)
	@if command -v uv >/dev/null 2>&1; then uv venv .venv -q && uv pip install -q --python .venv/bin/python torch numpy; \
	else python3 -m venv .venv && .venv/bin/pip install -q torch numpy; fi
	@echo -e "$(GREEN).venv ready.$(RESET)"

words: ## Download enable1 + common-30k into data/ (not committed; see data/README.md)
	@curl -sL -o data/enable1.txt https://raw.githubusercontent.com/dolph/dictionary/master/enable1.txt
	@curl -sL https://raw.githubusercontent.com/arstgit/high-frequency-vocabulary/master/30k.txt | tr -d '\r\t' | awk 'NF' > data/common-30k.txt
	@wc -l data/enable1.txt data/common-30k.txt

check-mps: ## Preflight: fail unless PyTorch gets a real GPU (MPS) in THIS shell - run before any long training
	@$(PYTHON) nano/device.py

data-smoke: ## ~2k boards -> data/wh_2k.npz
	@$(PYTHON) -m nano.data --boards 2000 --out data/wh_2k.npz

train-smoke: check-mps ## depth-4, 300 steps on data/wh_2k.npz -> runs/smoke_d4
	@$(PYTHON) -m nano.train --data data/wh_2k.npz --depth 4 --steps 300 --val-every 100 --out runs/smoke_d4

rollout-smoke: ## 20 unseen boards, student vs random swiper
	@$(PYTHON) -m nano.rollout --model runs/smoke_d4/model.pt --boards 20

data: ## PLAN.md scale: BOARDS boards -> DATA
	@$(PYTHON) -m nano.data --boards $(BOARDS) --out $(DATA)

train: check-mps ## Real run (one MPS trainer at a time): DEPTH, STEPS, DATA -> RUN
	@$(PYTHON) -m nano.train --data $(DATA) --depth $(DEPTH) --steps $(STEPS) --out $(RUN)

# ---------------------------------------------------------------------------
# Local vision Gemma 4 12B via mlx-vlm (Apple Silicon)
# ---------------------------------------------------------------------------
mlxvlm-up: ## Start the mlx-vlm OpenAI-compatible server for Gemma 4 12B (Apple Silicon)
	@echo -e "$(CYAN)Bringing up mlx-vlm server for '$(MLXVLM_MODEL)' on port $(MLXVLM_PORT)...$(RESET)"
	@if [ "$$(uname -s)" != "Darwin" ] || [ "$$(uname -m)" != "arm64" ]; then \
		echo -e "$(RED)Error: mlx-vlm requires macOS on Apple Silicon (arm64).$(RESET)"; \
		exit 1; \
	fi
	@if [ ! -d ".venv" ]; then \
		echo -e "$(RED)Error: Virtual environment (.venv) not found. Run 'make setup' first.$(RESET)"; \
		exit 1; \
	fi
	@if ! .venv/bin/python -c "import mlx_vlm" >/dev/null 2>&1; then \
		echo "  mlx-vlm not installed in .venv - installing (Apple Silicon only)..."; \
		.venv/bin/python -m pip install -U mlx-vlm; \
	fi
	@if curl -fsS "$(MLXVLM_HOST)/health" >/dev/null 2>&1; then \
		echo -e "  $(GREEN)mlx-vlm server already running at $(MLXVLM_HOST).$(RESET)"; \
	else \
		echo "  Starting 'mlx_vlm.server' in the background (first run downloads weights)..."; \
		nohup .venv/bin/python -m mlx_vlm.server --model "$(MLXVLM_MODEL)" --port $(MLXVLM_PORT) >/tmp/wordhunt-mlxvlm.log 2>&1 & \
		for i in $$(seq 1 60); do \
			if curl -fsS "$(MLXVLM_HOST)/health" >/dev/null 2>&1; then break; fi; \
			sleep 2; \
		done; \
		if ! curl -fsS "$(MLXVLM_HOST)/health" >/dev/null 2>&1; then \
			echo -e "$(RED)Error: mlx-vlm server did not come up. Check /tmp/wordhunt-mlxvlm.log.$(RESET)"; \
			exit 1; \
		fi; \
		echo -e "  $(GREEN)mlx-vlm server is up (logs: /tmp/wordhunt-mlxvlm.log).$(RESET)"; \
	fi
	@echo -e "$(GREEN)Ready: OpenAI-compatible endpoint at $(MLXVLM_HOST)/v1 (Gemma 4 12B, vision intact).$(RESET)"

mlxvlm-down: mlxvlm-stop ## Alias for mlxvlm-stop

mlxvlm-stop: ## Stop the mlx-vlm server process entirely
	@echo -e "$(YELLOW)Stopping the mlx-vlm server...$(RESET)"
	@pkill -f "mlx_vlm.server" 2>/dev/null && echo -e "  $(GREEN)mlx-vlm server stopped.$(RESET)" || echo -e "  $(YELLOW)No 'mlx_vlm.server' process found.$(RESET)"

mlxvlm-status: ## Show mlx-vlm server status and the model it is serving
	@echo -e "$(MAGENTA)--- mlx-vlm Status ---$(RESET)"
	@if curl -fsS "$(MLXVLM_HOST)/health" >/dev/null 2>&1; then \
		echo -e "Server: $(GREEN)running$(RESET) ($(MLXVLM_HOST))"; \
		curl -fsS "$(MLXVLM_HOST)/v1/models" 2>/dev/null || echo "  (could not read /v1/models)"; \
		echo ""; \
	else \
		echo -e "Server: $(YELLOW)not running$(RESET) (start with 'make mlxvlm-up')"; \
	fi
