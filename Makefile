# Word Hunt arena - game server, nano training, local Gemma understudy. Targets check-mps and mlxvlm-* are
# ported from actionfleet's Makefile (same model tag and flags).

.PHONY: help setup setup-server words serve dev smoke docker-build docker-run \
	check-mps data-smoke train-smoke rollout-smoke data train mlxvlm-up mlxvlm-down mlxvlm-stop mlxvlm-status gemma-seat

SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= .venv/bin/python

# Game server (wordhunt/server.py). Default port 8000 so it does not collide with mlx-vlm on 8080.
# HOST=0.0.0.0 so a phone on the same Wi-Fi can join via the LAN URL printed at startup.
HOST ?= 0.0.0.0
PORT ?= 8000
# Round knobs read by wordhunt/room.py; shorten for fast local rounds: make serve WH_COUNTDOWN_S=3 WH_RACE_S=20
WH_COUNTDOWN_S ?= 20
WH_RACE_S      ?= 75
export WH_COUNTDOWN_S WH_RACE_S
UVICORN_FLAGS := --ws-ping-interval 20 --ws-ping-timeout 20
WORDS := data/enable1.txt data/common-30k.txt
IMAGE ?= wordhunt:local

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

setup: setup-server ## Create .venv with server deps + torch + numpy (uv if present)
	@if command -v uv >/dev/null 2>&1; then uv pip install -q --python .venv/bin/python torch numpy; \
	else .venv/bin/pip install -q torch numpy; fi
	@echo -e "$(GREEN).venv ready (server + training).$(RESET)"

setup-server: ## Create .venv with just the game server deps (fastapi, uvicorn) - fast, no torch
	@if command -v uv >/dev/null 2>&1; then \
		[ -d .venv ] || uv venv .venv -q; \
		uv pip install -q --python .venv/bin/python -r requirements.txt; \
	else \
		[ -d .venv ] || python3 -m venv .venv; \
		.venv/bin/pip install -q -r requirements.txt; \
	fi
	@echo -e "$(GREEN).venv ready (server).$(RESET)"

words: ## Download enable1 + common-30k into data/ (not committed; see data/README.md)
	@curl -sL -o data/enable1.txt https://raw.githubusercontent.com/dolph/dictionary/master/enable1.txt
	@curl -sL https://raw.githubusercontent.com/arstgit/high-frequency-vocabulary/master/30k.txt | tr -d '\r\t' | awk 'NF' > data/common-30k.txt
	@wc -l data/enable1.txt data/common-30k.txt

# Word lists are gitignored; fetch them on first run so serve/dev/smoke work from a fresh clone.
$(WORDS):
	@$(MAKE) --no-print-directory words

# ---------------------------------------------------------------------------
# Game server (same uvicorn flags as the Dockerfile / Cloud Run)
# ---------------------------------------------------------------------------
serve: $(WORDS) ## Run the game server on HOST:PORT (default 0.0.0.0:8000); phones join via the LAN URL
	@if [ ! -x .venv/bin/uvicorn ]; then echo -e "$(RED)No server deps in .venv - run 'make setup-server' first.$(RESET)"; exit 1; fi
	@echo -e "$(CYAN)Word Hunt arena$(RESET)  local: $(GREEN)http://localhost:$(PORT)$(RESET)  LAN: $(GREEN)http://$$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $$1}' || echo '<lan-ip>'):$(PORT)$(RESET)"
	@echo -e "  round: countdown $(WH_COUNTDOWN_S)s, race $(WH_RACE_S)s   (override: make serve WH_COUNTDOWN_S=3 WH_RACE_S=20)"
	@.venv/bin/uvicorn wordhunt.server:app --host $(HOST) --port $(PORT) $(UVICORN_FLAGS)

dev: $(WORDS) ## Like serve, but auto-reloads on changes to wordhunt/ (rooms are in-memory and reset on reload)
	@if [ ! -x .venv/bin/uvicorn ]; then echo -e "$(RED)No server deps in .venv - run 'make setup-server' first.$(RESET)"; exit 1; fi
	@echo -e "$(CYAN)Word Hunt arena (reload)$(RESET)  $(GREEN)http://localhost:$(PORT)$(RESET)  round: $(WH_COUNTDOWN_S)s + $(WH_RACE_S)s"
	@.venv/bin/uvicorn wordhunt.server:app --host $(HOST) --port $(PORT) $(UVICORN_FLAGS) --reload --reload-dir wordhunt

smoke: $(WORDS) ## Boot the server on a scratch port, hit /api/health + /api/rooms, shut it down
	@if [ ! -x .venv/bin/uvicorn ]; then echo -e "$(RED)No server deps in .venv - run 'make setup-server' first.$(RESET)"; exit 1; fi
	@port=$$(.venv/bin/python -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])'); \
	log=$$(mktemp -t wordhunt-smoke); \
	.venv/bin/uvicorn wordhunt.server:app --host 127.0.0.1 --port $$port $(UVICORN_FLAGS) >"$$log" 2>&1 & pid=$$!; \
	trap 'kill $$pid 2>/dev/null; wait $$pid 2>/dev/null' EXIT; \
	for i in $$(seq 1 50); do curl -fsS "http://127.0.0.1:$$port/api/health" >/dev/null 2>&1 && break; sleep 0.2; done; \
	if ! health=$$(curl -fsS "http://127.0.0.1:$$port/api/health"); then \
		echo -e "$(RED)server did not come up; log:$(RESET)"; cat "$$log"; exit 1; fi; \
	echo "health:    $$health"; \
	code=$$(curl -fsS -X POST "http://127.0.0.1:$$port/api/rooms" | .venv/bin/python -c 'import json,sys;print(json.load(sys.stdin)["code"])'); \
	echo "room:      $$code"; \
	info=$$(curl -fsS "http://127.0.0.1:$$port/api/rooms/$$code"); \
	echo "snapshot:  $$info"; \
	curl -fsS -o /dev/null "http://127.0.0.1:$$port/" && curl -fsS -o /dev/null "http://127.0.0.1:$$port/r/$$code" && curl -fsS -o /dev/null "http://127.0.0.1:$$port/s/$$code"; \
	echo "index:     ok (/, /r/$$code, /s/$$code)"; \
	curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$$port/api/rooms/NOPE" | grep -q '^404$$' && echo "404:       ok (unknown room)"; \
	echo -e "$(GREEN)smoke passed.$(RESET)"

docker-build: ## Build the Cloud Run image locally (fetches word lists at build time)
	@docker build -t $(IMAGE) .

docker-run: ## Run the built image on PORT (default 8000), same entrypoint as Cloud Run
	@echo -e "$(CYAN)$(IMAGE)$(RESET) -> $(GREEN)http://localhost:$(PORT)$(RESET)"
	@docker run --rm -it -p $(PORT):8080 -e WH_COUNTDOWN_S=$(WH_COUNTDOWN_S) -e WH_RACE_S=$(WH_RACE_S) $(IMAGE)

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

# ---------------------------------------------------------------------------
# Gemma 12B seat: this Mac's mlx-vlm plays in a room (local or Cloud Run) over ws(s)
#   make gemma-seat ROOM=AB12 SERVER=https://wordhunt-xxxx.a.run.app
#   SEAT_ARGS="--modality image --filter-solver --rounds 1 --start" for variants
# ---------------------------------------------------------------------------
ROOM      ?=
SERVER    ?= ws://localhost:8000
SEAT_ARGS ?=

gemma-seat: ## Join room ROOM on SERVER as the Gemma 12B seat (needs mlxvlm-up); SEAT_ARGS for extras
	@if [ -z "$(ROOM)" ]; then echo -e "$(RED)Usage: make gemma-seat ROOM=CODE SERVER=https://...$(RESET)"; exit 1; fi
	@if ! $(PYTHON) -c "import openai, PIL, websockets" >/dev/null 2>&1; then \
		echo "  installing gemma_seat deps into .venv..."; \
		if command -v uv >/dev/null 2>&1; then uv pip install -q --python $(PYTHON) -r gemma_seat/requirements.txt; \
		else $(PYTHON) -m pip install -q -r gemma_seat/requirements.txt; fi; \
	fi
	@if ! curl -fsS "$(MLXVLM_HOST)/health" >/dev/null 2>&1; then \
		echo -e "$(RED)mlx-vlm is not running at $(MLXVLM_HOST); run 'make mlxvlm-up' first.$(RESET)"; exit 1; \
	fi
	@echo -e "$(CYAN)Gemma 12B seat -> room $(ROOM) on $(SERVER)$(RESET)"
	@MLXVLM_BASE_URL="$(MLXVLM_HOST)/v1" MLXVLM_MODEL="$(MLXVLM_MODEL)" \
		$(PYTHON) -m gemma_seat.bot --room "$(ROOM)" --server "$(SERVER)" $(SEAT_ARGS)

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
