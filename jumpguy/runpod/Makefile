# Jump Guy RunPod train hub.
# Mirrors actionfleet: make train-swm-runpod → make -C sandbox/runpod-train train
#
#   make -C jumpguy/runpod create   # needs RUNPOD_API_KEY
#   make -C jumpguy/runpod train    # sync + setup + collect/BC/PPO
#   make -C jumpguy/runpod logs
#   make -C jumpguy/runpod pull
#   make -C jumpguy/runpod watch
#   make -C jumpguy/runpod destroy  # pods do NOT auto-stop
#
# Identity: RUNPOD_SSH_IDENTITY or ~/.ssh/runpod_ed25519 (SWM) / ~/.ssh/runpod_nano
# JUMPGUY_REMOTE / JUMPGUY_SSH_KEY are filled from `runpodctl ssh info <id>`.

SCRIPTS := scripts

.PHONY: env create destroy down pod-down setup sync clone push-runs train logs pull watch smoke

env:
	@bash -c 'source $(SCRIPTS)/env.sh && jumpguy_print_env'

create:
	bash $(SCRIPTS)/create-pod.sh

destroy down pod-down:
	bash $(SCRIPTS)/destroy-pod.sh

setup:
	bash $(SCRIPTS)/setup-host.sh

sync clone:
	bash $(SCRIPTS)/sync-code.sh

push-runs:
	bash $(SCRIPTS)/push-runs.sh

train: sync setup
	bash $(SCRIPTS)/train.sh

logs:
	bash $(SCRIPTS)/logs.sh

pull:
	bash $(SCRIPTS)/pull-ckpt.sh

watch:
	bash $(SCRIPTS)/watch-train.sh

smoke:
	@bash -c 'source $(SCRIPTS)/env.sh && jumpguy_require_ssh && \
	  jumpguy_ssh "$$JUMPGUY_REMOTE" "cd $$JUMPGUY_REMOTE_DIR && python3 -m jumpguy device cuda && python3 -m jumpguy collect --episodes 4 --out /tmp/jg.npz && python3 -m jumpguy train --bc /tmp/jg.npz --steps 40 --out /tmp/jg_bc"'
